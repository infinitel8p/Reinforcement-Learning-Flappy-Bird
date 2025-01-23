import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import gymnasium
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the DQN network


class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay buffer for experience replay


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Epsilon-greedy policy for action selection


def epsilon_greedy_action(state, q_network, epsilon, action_space):
    if np.random.random() < epsilon:
        return action_space.sample()  # Explore
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()  # Exploit


# Initialize environment
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

# Reward history
reward_history = []

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

q_network = DQNetwork(input_size=state_space, output_size=action_space)
target_network = DQNetwork(input_size=state_space, output_size=action_space)
target_network.load_state_dict(q_network.state_dict())  # Sync weights

replay_buffer = ReplayBuffer()

# ! Changed learning rate from 0.001 to 0.005
optimizer = optim.Adam(q_network.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

# Hyperparameters
num_episodes = 500
batch_size = 128  # ! Changed batch size from 64 to 128
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
update_target_steps = 50  # ! Changed update target steps from 100 to 50

# Load previous model if it exists
try:
    q_network.load_state_dict(torch.load("flappy_dqn.pth"))
    print("Loaded existing model weights.")
except FileNotFoundError:
    print("No previous model found, starting fresh.")

# Load reward history if it exists
try:
    with open("reward_history.pkl", "rb") as f:
        reward_history = pickle.load(f)
        print("Loaded existing reward history.")
except FileNotFoundError:
    print("No previous reward history found, starting fresh.")

# Load epsilon value if it exists
try:
    with open("epsilon.pkl", "rb") as f:
        epsilon = pickle.load(f)
        print(f"Loaded previous epsilon: {epsilon}")
except FileNotFoundError:
    print("No previous epsilon found, starting fresh.")

try:
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            # Select action
            action = epsilon_greedy_action(
                state, q_network, epsilon, env.action_space)

            # Take action and observe result
            next_state, reward, terminated, _, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            # Add experience to replay buffer
            replay_buffer.add((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            if terminated:
                break

            # Train the network
            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Q-learning target
                q_values = q_network(states)
                next_q_values = target_network(next_states)
                target_q_values = rewards + gamma * \
                    torch.max(next_q_values, dim=1)[0] * (1 - dones)

                # Loss calculation
                current_q_values = q_values.gather(
                    1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(current_q_values, target_q_values.detach())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network
        if episode % update_target_steps == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Save periodically
        if episode % 50 == 0:
            torch.save(q_network.state_dict(), "flappy_dqn.pth")
            with open("reward_history.pkl", "wb") as f:
                pickle.dump(reward_history, f)
            with open("epsilon.pkl", "wb") as f:
                pickle.dump(epsilon, f)
            print(
                f"Saved model, reward history, and epsilon at episode {episode}.")

        print(f"Episode {episode}, Epsilon: {
              epsilon:.2f}, Total Reward: {total_reward:.2f}")
        reward_history.append(total_reward)

finally:
    env.close()
    print("Environment closed.")

# Final save
torch.save(q_network.state_dict(), "flappy_dqn.pth")
with open("reward_history.pkl", "wb") as f:
    pickle.dump(reward_history, f)
with open("epsilon.pkl", "wb") as f:
    pickle.dump(epsilon, f)
print("Final model, epsilon and reward history saved.")

# Plot rewards
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.show()
