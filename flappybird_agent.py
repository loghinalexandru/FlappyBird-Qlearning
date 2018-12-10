import pygame
import numpy as np
import keras
from ple.games.flappybird import FlappyBird
from ple import PLE
from keras.models import Sequential
from keras.layers import InputLayer, Dense , Dropout

def get_features(state):
    features_list = []
    features_list.append(state["player_y"])
    features_list.append(state["player_vel"])
    features_list.append(state["player_y"] - (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2)
    return np.array(features_list).reshape(1,3)

def build_nn():
    model = Sequential()
    model.add(InputLayer((3,))) # player_y , velocity , distance from player to pipe gap ,
    model.add(Dense(50 , activation="relu"))
    model.add(Dense(50 , activation="relu"))
    model.add(Dense(2 , activation="sigmoid"))
    model.compile(keras.optimizers.Adam(lr = 0.1), loss = "mse", metrics = ["accuracy"])
    return model

if __name__ == "__main__":
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True, force_fps=False)
    env.display_screen = True
    env.force_fps = True
    env.init()
    model = build_nn()
    while True:
        if env.game_over():
            env.reset_game()
        prediction = model.predict(get_features(game.getGameState())).reshape(2,)
        if(prediction[0] > prediction[1]):
            env.act(0)
        else:
            env.act(119)
