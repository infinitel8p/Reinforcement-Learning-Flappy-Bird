# References:
# Agent + training: https://github.com/nathanwbailey/flappy_bird_reinforcement_learning
# pyGame learning environment: https://github.com/ntasfi/PyGame-Learning-Environment
# pyGame recorder: https://github.com/tdrmk/pygame_recorder

import os
import sys
sys.path.append("./pygame-le")
import pygame as pg
from ple import PLE 
from ple.games.flappybird import FlappyBird
import agent
import matplotlib.pyplot as plt
import torch
import time
import argparse


class GraphSaver():
    """Class to save the graphs and the network weights.
    """
    def __init__(self, plot_savedir: str, net_savedir: str):
        """Initialize the GraphSaver class.

        Args:
            plot_savedir (str): Save directory for the plots
            net_savedir (str): Save directory for the network weights
        """

        self.plot_savedir = plot_savedir
        self.net_savedir = net_savedir
        self.init_save_time = time.strftime('%H.%M.%S-%d.%m.%Y')

    def plot_single_graph(self, subplot, values, y_label: str, plot_avg=True):
        """Plot a single graph.
        Args:
            subplot (List[3]): A list of integers specifying the subplot configuration (e.g., [1, 2, 1] for 1 row, 2 columns, first plot).
            values (torch.Tensor): A tensor containing the values to be plotted.
            y_label (str): Label for the y-axis.
            plot_avg (bool, optional): Whether to plot the average of the values. Defaults to True.

        Example:
            self.plot_single_graph((2,2,1), durations_t, 'Duration')
        """

        plt.subplot(*subplot)
        plt.ylabel(y_label)
        plt.plot(values.numpy(), label=y_label)
        if len(values) >= 100 and plot_avg:
            means = values.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="Avg (100)")
        plt.legend()

    def plot_graphs(self, agent):
        """Plot the graphs for the agent.

        Args:
            agent (Agent): Agent to plot the graphs for
        """

        plt.figure(figsize=(12, 8))
        plt.clf()

        # Get data from the agent
        durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(agent.episode_rewards, dtype=torch.float)
        epsilons_t = torch.tensor(agent.episode_epsilons, dtype=torch.float)
        scores_t = torch.tensor(agent.episode_scores, dtype=torch.float)

        # Plot the graphs
        self.plot_single_graph((2,2,1), durations_t, 'Duration')
        self.plot_single_graph((2,2,2), rewards_t, 'Reward')
        self.plot_single_graph((2,2,3), scores_t, 'Score')
        self.plot_single_graph((2,2,4), epsilons_t, 'Epsilon', plot_avg=False)

        # Save the plot
        plt.tight_layout()
        plt.pause(0.001)  # Pause to update the plots
        plt.savefig(os.path.join(self.plot_savedir, agent.network_type + f'_{self.init_save_time}_training.png'))
        plt.close()

    def save_net(self, agent):
        """ Save the network weights for the agent.

        Args:
            agent (Agent): Agent to save the network weights for
        """

        dir_path = os.path.join(self.net_savedir, f'weights_{self.init_save_time}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(agent.target_net.state_dict(), os.path.join(dir_path, f'{agent.network_type}_target_net.pt'))
        torch.save(agent.policy_net.state_dict(), os.path.join(dir_path, f'{agent.network_type}_policy_net.pt'))


if __name__ == "__main__":
    """Main function to run the Flappy Bird game.
    """

    parser = argparse.ArgumentParser(description='Train or test model. Default is training a new model.')

    parser.add_argument('-t', '--test', action='store_true', help='Test our model', dest='test')
    parser.add_argument('-e', '--episodes', type=int, help='Number of episodes to train the model. Defaults to 50', default=50)
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (⚠️ Not properly tested)', dest='headless')
    parser.add_argument('--no-recording', action='store_true', help='Do not record the game', dest='no_recording')
    args = parser.parse_args()

    if args.headless:
        # run in headless mode
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = FlappyBird(width=256, height=256)
    p = PLE(game, display_screen=True)
    p.init()
    actions = p.getActionSet()

    # List of possible actions is go up or do nothing
    action_dict = {0: actions[1], 1: actions[0]}

    # get the initial game state
    state = p.getGameState()
    len_state = len(state)
    n_actions = len(action_dict)

    print("Initialized game")
    flappy_bird_plot_saver = GraphSaver('graphs', 'weights')

    # Create the agent and train it
    fb_agent = agent.Agent(
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
        TAU=0.005,
        network_type='DuelingDQN',
        lr=1e-4,
        graph_saver=flappy_bird_plot_saver,
        headless=args.headless,
        recording=not args.no_recording
    )

    if args.test:
        policy_net_weights = torch.load('weights/DuelingDQN_policy_net.pt')
        target_net_weights = torch.load('weights/DuelingDQN_target_net.pt')
        fb_agent.policy_net.load_state_dict(policy_net_weights)
        fb_agent.target_net.load_state_dict(target_net_weights)
        fb_agent.eps = 0.05
    
    fb_agent.train(episodes=args.episodes, env=p)