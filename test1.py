import os
import time
import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import gymnasium
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from itertools import count
import random

savedir = 'medium_tutorial'
start_time = time.strftime('%H:%M:%S-%d.%m.%Y')

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, network_type='DQN', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.layer1 = nn.Linear(input_dim,64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 256) 
        self.layer5 = nn.Linear(256, 256)
        # # Dueling DQN predicts both the value of the state and the advantage of each possible action
        # # Best action should have advantage of 0
        ## Outputs are combined to generate the Q values

        if network_type == 'DuelingDQN':
            self.inner_state_values = nn.Linear(256, 256)
            self.state_values = nn.Linear(256,1)

            self.inner_advantages = nn.Linear(256, 256)
            self.advantages = nn.Linear(256, output_dim)
        else:
            self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        x = F.relu6(self.layer5(x))
        if self.network_type == 'DuelingDQN':
            inner_state_values = F.relu6(self.inner_state_values(x))
            state_values = self.state_values(inner_state_values)

            inner_advantages = F.relu6(self.inner_advantages(x))
            advantages = self.advantages(inner_advantages)
            output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
            return output
        else:
            return self.output(x)


'''
MemoryRecall module, this is needed to cache data from the environment.

Then to randomly sample a batch when needed for training

'''

class MemoryRecall():
    def __init__(self, memory_size) -> None:
        self.memory_size=memory_size
        self.memory = deque(maxlen = self.memory_size)
    
    def cache(self, data):
        self.memory.append(data)

    def recall(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Agent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU, network_type='DDQN') -> None:
        #Set all the values up
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim = action_dim
        #self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        #Select the GPU if we have one
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_rewards = []
        #Create the cache recall memory
        self.cache_recall = MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type
        #Create the dual Q networks - target and policy nets
        self.policy_net = DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)
        self.target_net = DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type).to(self.device)
        #No need to calculate gradients for target net parameters as these are periodically copied from the policy net
        for param in self.target_net.parameters():
            param.requires_grad = False
        #Copy the initial parameters from the policy net to the target net to align them at the start
        #Diff
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0
    
    #We want to use no gradient computation, as we do not update the networks paramaters
    @torch.no_grad()
    def take_action(self, state):
        #Decay and cap the epsilon value
        self.eps = self.eps*self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        # self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY_VALUE)
        #Take a greedy action
        if self.eps < np.random.rand():
            state = state[None, :]
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        #Else take a random action
        else:
            action_idx = random.randint(0, self.action_dim-1)
        self.steps_done += 1
        return action_idx

    
    def plot_rewards(self):
        plt.figure(1)
        plt.clf()
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        #Plot the rewards
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages of the rewards and plot them too, to show a running average on the graph
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig(f'{savedir}/training_{self.network_type}_{start_time}.png')
    
    #Function to copy the policy net parameters to the target net
    # def update_target_network(self):
    #     self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        #Update the parameters in the target network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        #Only pop the data if we have enough for the network
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        #Grab the batch
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        #reform the batch so we can grab the states, actions etc easily
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])
        #batch[1] gives us the next_state after the action, we want to create a mask to filter out the states where we end the run (i.e. the flappy bird dies). the end of the run will give a state of None which cannot be inputted into the network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[1])), device=self.device, dtype=torch.bool)
        #Grab the next states that are not final states
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        #Grab the action and the reward
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])
        next_state_action_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        #Get the Q values from the policy network and then get the Q value for the given action taken
        #[32,1]
        state_action_values = self.policy_net(state).gather(1, action)
        # print(self.target_net(non_final_next_states).max(1, keepdim=True)[0].size())
        #Use the target network to get the maximum Q for the next state across all the actions
        with torch.no_grad():
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        #Calcuate the expected state action values as the per equation we use
        expected_state_action_values = (next_state_action_values * self.GAMMA) + reward
        #Using the L1 Loss, calculate the difference between the expected and predicted values and optimize the policy network only
        loss_fn = torch.nn.SmoothL1Loss()
        # print(state_action_values.size())
        #print(expected_state_action_values.size())
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes, env):
        self.steps_done = 0
        for episode in range(episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            #state = torch.tensor(list(state.values()), dtype=torch.float32, device=self.device)
            #Inf count functiom
            total_reward = 0
            for c in count():
                #Choose an action, get back the reward and the next state as a result of taking the action
                action = self.take_action(state) # 0 or 1
                obs, reward, terminated, _, info = env.step(action)
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action], device=self.device)
                next_state = torch.tensor(obs, dtype=torch.float32, device=self.device)
                #next_state = torch.tensor(list(next_state.values()), dtype=torch.float32, device=self.device)
                done = terminated
                #if game is over, next state is None
                if done:
                    next_state = None
                #Cache a tuple of the date
                self.cache_recall.cache((state, next_state, action, reward, done))
                #Set the state to the next state
                state = next_state
                #Optimize the model and update the target network 
                self.optimize_model()
                self.update_target_network()
                if done:
                    #Update the number of rewards for the episode
                    self.episode_rewards.append(total_reward)
                    if episode % 20 == 0:
                        #Plot them and save the networks
                        self.plot_rewards()
                    
                    print(f'Episode: {episode} | EPS: {self.eps: .2f} | Reward: {total_reward : .2f} | Durations: {c+1} | Score: {info["score"]}')
                    
                    if episode % 100 == 0:
                        torch.save(self.target_net.state_dict(), f'{savedir}/target_net_{self.network_type}_{start_time}.pt')
                        torch.save(self.policy_net.state_dict(), f'{savedir}/policy_net_{self.network_type}_{start_time}.pt')
                    #Start a new episode
                    break


env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=True)
obs, _ = env.reset()
n_actions = 2
agent = Agent(
    BATCH_SIZE=32,
    MEMORY_SIZE=100000,
    GAMMA=0.99,
    input_dim=len(obs),
    output_dim=n_actions,
    action_dim=n_actions,
    action_dict=None, # is not needed
    EPS_START=1.0,
    EPS_END=0.05,
    EPS_DECAY_VALUE=0.999995,
    TAU = 0.005,
    network_type='DuelingDQN',
    lr = 1e-4
)

agent.train(episodes=50000, env=env)