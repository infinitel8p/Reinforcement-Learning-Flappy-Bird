import os
import sys
# medium_tutorial_full_code/flappy_bird_reinforcement_learning/PyGame-Learning-Environment/PyGame-Learning-Environment/ple
sys.path.append("medium_tutorial_2/pygame-le")
import pygame as pg
from ple import PLE 
from ple.games.flappybird import FlappyBird
from ple import PLE
import agent

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

#Create the agent and train it
agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99, input_dim=len_state, output_dim=n_actions, action_dim=n_actions, action_dict=action_dict, EPS_START=1.0, EPS_END=0.05, EPS_DECAY_VALUE=0.999995, TAU = 0.005, network_type='DuelingDQN', lr = 1e-4)

agent.train(episodes=100, env=p)

