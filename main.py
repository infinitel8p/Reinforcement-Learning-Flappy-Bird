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
    def __init__(self, max_size=50000):
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


# Create save directory if it doesn't exist
save_dir = "savefile"
os.makedirs(save_dir, exist_ok=True)

# Initialize environment
# render_mode="human" for visualization, None for faster training
env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=True)

# Reward history and stats
reward_history = []
episode_stats = {"max_reward": [], "average_reward": []}

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

q_network = DQNetwork(input_size=state_space, output_size=action_space)
target_network = DQNetwork(input_size=state_space, output_size=action_space)
target_network.load_state_dict(q_network.state_dict())  # Sync weights

replay_buffer = ReplayBuffer()

# ! Changed learning rate from 0.001 to 0.005
optimizer = optim.Adam(q_network.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = nn.SmoothL1Loss()

# Hyperparameters
num_episodes = 500
batch_size = 128
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
update_target_steps = 50  # ! Changed update target steps from 100 to 50

# Load previous model if it exists
try:
    with open(os.path.join(save_dir, "training_state.pkl"), "rb") as f:
        training_state = pickle.load(f)

    # Restore the data
    q_network.load_state_dict(training_state["q_network_state_dict"])
    reward_history = training_state["reward_history"]
    epsilon = training_state["epsilon"]
    start_episode = training_state["episode_counter"]
    episode_stats = training_state["episode_stats"]

    print(f"Loaded training state from episode {start_episode}.")
except FileNotFoundError:
    # Default initialization if no saved state exists
    print("No training state file found, starting fresh.")
    start_episode = 0
    reward_history = []
    epsilon = 1.0
    episode_stats = {"max_reward": [], "average_reward": []}

try:
    for episode in range(start_episode, num_episodes+start_episode):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        state = np.clip(state, env.observation_space.low,
                        env.observation_space.high)
        done = False
        total_reward = 0

        while not done:
            # Select action
            action = epsilon_greedy_action(
                state, q_network, epsilon, env.action_space)

            # Take action and observe result
            next_state, reward, terminated, _, info = env.step(action)

            # Convert next_state to float32 and clip it to observation_space bounds
            next_state = np.array(next_state, dtype=np.float32)
            next_state = np.clip(
                next_state, env.observation_space.low, env.observation_space.high)

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

                states = np.array(states, dtype=np.float32)
                states = torch.tensor(states)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = np.array(next_states, dtype=np.float32)
                next_states = torch.tensor(next_states)
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

        # Update epsilon with periodic restart
        if episode % 500 == 0 and epsilon == min_epsilon:  # Restart exploration every 500 episodes
            epsilon = 0.1  # Restart to a higher exploration rate
        else:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network
        if episode % update_target_steps == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Track rewards and stats
        reward_history.append(total_reward)
        episode_stats["max_reward"].append(max(reward_history))
        episode_stats["average_reward"].append(np.mean(reward_history))

        # Save periodically
        if episode % 50 == 0:
            training_state = {
                "q_network_state_dict": q_network.state_dict(),
                "reward_history": reward_history,
                "epsilon": epsilon,
                "episode_counter": episode + 1,  # Save the next episode
                "episode_stats": episode_stats,
            }

            with open(os.path.join(save_dir, "training_state.pkl"), "wb") as f:
                pickle.dump(training_state, f)

            print(f"Saved training state at episode {episode}.")

        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")

        print(f"Episode {episode}, Epsilon: {
              epsilon:.2f}, Total Reward: {total_reward:.2f}")

finally:
    env.close()
    print("Environment closed.")

# Final save
training_state = {
    "q_network_state_dict": q_network.state_dict(),
    "reward_history": reward_history,
    "epsilon": epsilon,
    "episode_counter": episode + 1,
    "episode_stats": episode_stats,
}

with open(os.path.join(save_dir, "training_state.pkl"), "wb") as f:
    pickle.dump(training_state, f)

print(f"Saved final training state at episode {episode + 1}.")


def moving_average(data, window_size=10):
    """Smoothing function

    Args:
        data (list): List of data points
        window_size (int, optional): Size of the window. Defaults to 10.

    Returns:
        list: Smoothed data
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Plotting the training progress
smoothed_rewards = moving_average(reward_history, window_size=10)
plt.plot(reward_history, label="Raw Rewards")
plt.plot(range(len(smoothed_rewards)),
         smoothed_rewards, label="Smoothed Rewards")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.savefig(os.path.join(save_dir, f"final_plot_smoothed_{
            time.strftime('%H-%M-%S-%d.%m.%Y')}.png"))
plt.show()
