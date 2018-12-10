from ple.games.flappybird import FlappyBird
import pygame
from ple import PLE
import numpy as np
from random import random
import time

def processState(state):
    ans = list()
    ans.append(state["player_vel"])
    ans.append(state["player_y"] - (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) * 0.5)
    ans.append(state["next_pipe_dist_to_player"])
    return np.array(ans).reshape((1, 3))


game = FlappyBird()
env = PLE(game, fps=30,state_preprocessor=processState , display_screen=True, force_fps=False)
env.display_screen = True
env.force_fps = True
env.init()

while True:
    if env.game_over():
        env.reset_game()
    state = env.getGameState()
    last = time.clock()
    for event in pygame.event.get():
        if(event.key == pygame.K_SPACE):
            env.act(119)
        if(event.key == pygame.K_a):
            env.act(0)
