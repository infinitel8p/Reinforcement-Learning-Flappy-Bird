import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
import itertools
from experience_replay import ReplayMemory
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', "r") as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

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

    def run(self, is_training: bool = True, render: bool = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make(
            "CartPole-v1", render_mode="human" if render else None)

        # Create a DQN instance
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy = DQN(num_states, num_actions).to_device(device)

        rewards_per_episode = []

        # Define memory for Experience Replay if in training mode
        if is_training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                # Update the reward
                episode_reward += reward

                # Update the memory
                if is_training:
                    memory.append(
                        (state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

            rewards_per_episode.append(episode_reward)
