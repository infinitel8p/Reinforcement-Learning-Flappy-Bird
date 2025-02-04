import os
import sys
# medium_tutorial_full_code/flappy_bird_reinforcement_learning/PyGame-Learning-Environment/PyGame-Learning-Environment/ple
sys.path.append("medium_tutorial_2/pygame-le")
import pygame as pg
from ple import PLE 
from ple.games.flappybird import FlappyBird
from ple import PLE
import agent
import matplotlib.pyplot as plt
import torch
import time

game = FlappyBird(width=256, height=256)
p = PLE(game, display_screen=False)
p.init()
actions = p.getActionSet()
#List of possible actions is go up or do nothing
action_dict = {0: actions[1], 1: actions[0]}

#get the initial game state
state = p.getGameState()
print(state)
len_state = len(state)
n_actions = len(action_dict)

class GraphSaver():
    def __init__(self, plot_savedir, net_savedir):
        self.plot_savedir = plot_savedir
        self.net_savedir = net_savedir
        self.init_save_time = time.strftime('%H:%M:%S-%d.%m.%Y')

    def plot_single_graph(self, subplot, values, y_label, plot_avg=True):
        plt.subplot(*subplot)
        plt.ylabel(y_label)
        plt.plot(values.numpy(), label=y_label)
        if len(values) >= 100 and plot_avg:
            means = values.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="Avg (100)")
        plt.legend()

    def plot_graphs(self, agent):
        plt.figure(figsize=(12, 8))
        plt.clf()

        # Get data from the agent
        durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(agent.episode_rewards, dtype=torch.float)
        epsilons_t = torch.tensor(agent.episode_epsilons, dtype=torch.float)
        scores_t = torch.tensor(agent.episode_scores, dtype=torch.float)

        self.plot_single_graph((2,2,1), durations_t, 'Duration')
        self.plot_single_graph((2,2,2), rewards_t, 'Reward')
        self.plot_single_graph((2,2,3), scores_t, 'Score')
        self.plot_single_graph((2,2,4), epsilons_t, 'Epsilon', plot_avg=False)

        plt.tight_layout()
        plt.pause(0.001)  # Pause to update the plots
        plt.savefig(os.path.join(self.plot_savedir, agent.network_type + f'_{self.init_save_time}_training.png'))

    def save_net(self, agent):
        dir_path = os.path.join(self.net_savedir, f'weights_{self.init_save_time}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(agent.target_net.state_dict(), os.path.join(dir_path, f'{agent.network_type}_target_net.pt'))
        torch.save(agent.policy_net.state_dict(), os.path.join(dir_path, f'{agent.network_type}_policy_net.pt'))

flappy_bird_plot_saver = GraphSaver('medium_tutorial_2/graphs', 'medium_tutorial_2/weights')

#Create the agent and train it
agent = agent.Agent(
    BATCH_SIZE=32, 
    MEMORY_SIZE=100000, 
    GAMMA=0.99, 
    input_dim=len_state, 
    output_dim=n_actions, 
    action_dim=n_actions, 
    action_dict=action_dict, 
    EPS_START=1.0, 
    EPS_END=0.05, 
    EPS_DECAY_VALUE=0.999995, 
    TAU = 0.005, 
    network_type='DuelingDQN', 
    lr = 1e-4,
    graph_saver=flappy_bird_plot_saver
)

agent.train(episodes=109, env=p)

