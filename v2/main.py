import gymnasium
import flappy_bird_gymnasium

import random
import torch
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

DATE_FORMAT = "%H:%M:%S_%d-%m-%Y"
SAVEFILES_DIR = "savefile"
os.makedirs(SAVEFILES_DIR, exist_ok=True)
matplotlib.use('Agg')


class Agent:
    def __init__(self, hyperparameter_set):
        with open('./hyperparameters.yml', "r") as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters
        # Environment ID (FlappyBird-v0 or CartPole-v1)
        self.env_id = hyperparameters["env_id"]
        # Replay memory size
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        # Mini-batch size (training data set sampled from the replay memory)
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        # Initial epsilon value for epsilon-greedy policy (1 = 100% random actions)
        self.epsilon_initial = hyperparameters["epsilon_initial"]
        # Epsilon decay rate
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        # Minimum epsilon value
        self.epsilon_min = hyperparameters["epsilon_min"]
        # Rate at which the policy network is synced with the target network
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        # Learning rate for the Adam optimizer (alpha)
        self.learning_rate_adam = hyperparameters["learning_rate_adam"]
        # Discount factor for the Q-value estimation (gamma)
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        # Stop training when the reward exceeds this value
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        # Number of nodes in the first fully connected layer
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        # Optional environment parameters
        self.env_make_params = hyperparameters.get("env_make_params", {})

        self.loss_fn = torch.nn.MSELoss()  # NN loss function (Mean Squared Error)
        self.optimizer = None  # NN optimizer

        self.LOG_FILE = os.path.join(
            SAVEFILES_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(
            SAVEFILES_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(
            SAVEFILES_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training: bool = True, render: bool = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(
                DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make(
            "CartPole-v1", render_mode="human" if render else None)

        # Number of possible actions
        num_actions = env.action_space.n
        # Get observation space size
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []

        # Create policy DQN
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        # Define memory for Experience Replay if in training mode
        if is_training:
            # Epsilon value for epsilon-greedy policy
            epsilon = self.epsilon_initial

            # Initialize the replay memory
            memory = ReplayMemory(maxlen=self.replay_memory_size)

            # Create target DQN as a copy of the policy DQN
            target_dqn = DQN(num_states, num_actions,
                             self.fc1_nodes).to(device)
            # Copy the weights from policy DQN to target DQN
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer (Adam optimizer)
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_adam)

            # Track the number of steps taken to sync the policy with the target network
            step_count = 0

            epsilon_history = []

            best_reward = -np.inf

        else:
            # Load the trained model and switch to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(
                        action, dtype=torch.int64, device=device)
                else:
                    # turn off gradients for the inference since we are not training the model
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(
                            dim=0)).squeeze().argmax()

                # Execute the action
                new_state, reward, terminated, _, info = env.step(
                    action.item())

                # Update the reward
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(
                    new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Update the memory
                if is_training:
                    memory.append(
                        (state, action, new_state, reward, terminated))
                    step_count += 1

                # Move to the next state
                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({
                        (episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay,
                                  self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # for state, action, new_state, reward, terminated in mini_batch:
        #     if terminated:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target = reward + self.discount_factor_g * \
        #                 target_dqn(new_state).max()

        # Transpose the list of experiences and separate them into individual tensors
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + \
                (1-terminations) * self.discount_factor_g * \
                target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # clear the gradients
        loss.backward()  # compute the gradients (backpropagation)
        self.optimizer.step()  # update network parameters (weights, biases)


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
