import os
import torch
import model
import MemoryRecall
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import re
import time
from pygame_recorder import ScreenRecorder

savedir = 'recordings'
os.makedirs(savedir, exist_ok=True)
start_time = time.strftime('%H:%M:%S-%d.%m.%Y')

class Agent():
    """
    The agent class used to interact with the environment and train the model.
    It takes actions based on the epsilon greedy policy and trains the model using the DQN algorithm.
    """

    def __init__(self, BATCH_SIZE: int, MEMORY_SIZE: int, GAMMA: float, input_dim: int, output_dim: int, action_dim: int, action_dict: dict, EPS_START: float, EPS_END: float, EPS_DECAY_VALUE: float, lr: float, TAU: float, network_type: str = 'DDQN', graph_saver=None, device=None, headless: bool = False) -> None:
        """Initialize the agent with the parameters needed to train the model

        Args:
            BATCH_SIZE (int): The batch size to use for training
            MEMORY_SIZE (int): The size of the memory recall to use
            GAMMA (float): The discount factor for the rewards
            input_dim (int): The input dimensions of the model
            output_dim (int): The output dimensions of the model
            action_dim (int): The number of actions the agent can take
            action_dict (dict): The dictionary of actions the agent can take
            EPS_START (float): The starting epsilon value for the agent
            EPS_END (float): The ending epsilon value for the agent
            EPS_DECAY_VALUE (float): The decay value for epsilon
            lr (float): The learning rate for the model
            TAU (float): The TAU value for updating the target network
            network_type (str, optional): The type of network to use. Defaults to 'DDQN'.
            graph_saver (optional): The graph saver object to save the graphs. Defaults to None.
            device (optional): The device to use for training. Defaults to None.
            headless (bool, optional): Whether to run the game in headless mode. Defaults to False.
        """

        # Set all the values up
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        self.graph_saver = graph_saver
        self.headless = headless

        #Select the GPU if we have one
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            self.device = device
        print(f"Device: {self.device}")

        self.episode_durations = []
        self.episode_rewards = []
        self.episode_epsilons = []
        self.episode_scores = []

        #Create the cache recall memory
        self.cache_recall = MemoryRecall.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type

        #Create the dual Q networks - target and policy nets
        self.policy_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)
        self.target_net = model.DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)

        #No need to calculate gradients for target net parameters as these are periodically copied from the policy net
        for param in self.target_net.parameters():
            param.requires_grad = False

        #Copy the initial parameters from the policy net to the target net to align them at the start
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0
        self.FPS = 30
        self.clock = pg.time.Clock()
        self.recorder = ScreenRecorder(256, 256, self.FPS, os.path.join(savedir, f"{os.path.basename(re.sub(r'[<>:"/\\|?*]', '_', start_time))}.avi"))


    @torch.no_grad()
    def take_action(self, state):
        """ Take an action based on the epsilon greedy policy.
        Uses 'no gradient computation' as we do not update the networks parameters.

        Args:
            state (Tensor|None): The state to take the action on    

        Returns:
            The action to take
        """
        #Decay and cap the epsilon value
        self.eps = self.eps*self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        
        # Take an action based on the epsilon greedy policy
        if self.eps < np.random.rand():
            # Take the best action
            state = state[None, :]
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        else:
            # Take a random action
            action_idx = random.randint(0, self.action_dim-1)

        self.steps_done += 1
        return action_idx

    
    def plot_durations(self):
        """
        Plot the durations of the episodes
        """
        if not self.headless:
            self.graph_saver.plot_graphs(self)

    def update_target_network(self):
        """
        Update the target network using the policy network
        """

        # Get the state dicts for the target and policy networks
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Update the parameters in the target network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        """
        Optimize the model using the DQN algorithm
        """

        #Only pop the data if we have enough for the network
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        
        #Grab the batch
        batch = self.cache_recall.recall(self.BATCH_SIZE)

        #reform the batch so we can grab the states, actions etc easily
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])

        # batch[1] gives us the next_state after the action, so create a mask to filter out the states where we end the run (i.e. the flappy bird dies). 
        # The end of the run will give a state of None which cannot be inputted into the network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[1])), device=self.device, dtype=torch.bool)

        #Grab the next states that are not final states
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])

        #Grab the action and the reward
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])
        next_state_action_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)

        # Get the predicted Q values for the current state and action 
        state_action_values = self.policy_net(state).gather(1, action)

        # Get the predicted maximum Q values for the next state across all next actions
        with torch.no_grad():
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Calcuate the expected state action values as the per equation
        expected_state_action_values = (next_state_action_values * self.GAMMA) + reward

        # Using the L1 Loss, calculate the difference between the expected and predicted values and optimize the policy network only
        loss_fn = torch.nn.SmoothL1Loss()

        # Calculate the loss and optimize the model
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes: int, env):
        """Train the model

        Args:
            episodes (int): The number of episodes to train the model for
            env (PLE FlappyBird): The environment to train the model in
        """

        self.steps_done = 0

        for episode in range(episodes):
            reward_sum = 0
            env.reset_game()
            state = env.getGameState()
            state = torch.tensor(list(state.values()),
                                 dtype=torch.float32, device=self.device)
            start_time = time.time()

            #Inf count functiom
            for c in count():

                #Choose an action, get back the reward and the next state as a result of taking the action
                action = self.take_action(state)
                reward = env.act(self.action_dict[action])
                reward = torch.tensor([reward], device=self.device)
                reward_sum += reward.item()
                action = torch.tensor([action], device=self.device)
                next_state = env.getGameState()
                next_state = torch.tensor(list(next_state.values()), dtype=torch.float32, device=self.device)
                done = env.game_over()

                # if game is over, next state is None
                if done:
                    next_state = None

                #Cache a tuple of the date
                self.cache_recall.cache((state, next_state, action, reward, done))

                #Set the state to the next state
                state = next_state

                #Optimize the model and update the target network 
                self.optimize_model()
                self.update_target_network()
                pg.display.update()
                self.recorder.capture_frame(env.getScreen())

                if done:
                    #Update the number of durations for the episode
                    self.episode_durations.append(c+1)
                    self.episode_epsilons.append(self.eps)
                    self.episode_scores.append(env.score())
                    self.episode_rewards.append(reward_sum)
                    
                    #Plot them and save the networks
                    print(f'episode: {episode} | eps: {self.eps} | duration: {c+1} | reward: {reward_sum:.3f} | running time: {(time.time() - start_time):.3f}')
                    
                    if reward_sum > 1300:
                        print('Solved!')
                        self.graph_saver.save_net(self)
                        self.recorder.end_recording()
                        return
                    if episode % 20 == 0 and not self.headless:
                            self.graph_saver.plot_graphs(self)
                    if episode % 100 == 0:
                        self.graph_saver.save_net(self)

                    #Start a new episode
                    break

        self.recorder.end_recording()



