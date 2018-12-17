import pygame
import numpy as np
import keras
import random
from ple.games.flappybird import FlappyBird
from ple import PLE
from keras.models import Sequential
from keras.layers import InputLayer, Dense , Dropout

eps = 0.1
moves = [0, 119]
SIZE = 100000

def get_move(model, state):
	"""
	Generate moves
	
	Parameters
	----------
	model : object
	    NN model
	state : list
	    Features list
	
	Returns
	-------
	int
	    Index of next move (0 - None , 1 - Up)
	"""
	if np.random.random() < eps:
		return np.random.choice([0, 1])
	else:
		return np.argmax(model.predict(state).reshape(2,))

def get_features(state):
	"""
	Filter the game state for certain features
	
	Parameters
	----------
	state : list
	    Game state
	
	Returns
	-------
	list
	    Selected features
	"""
	features_list = []
	features_list.append(state["player_vel"])
	features_list.append(state["player_y"] -(state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2)
	features_list.append(state["next_pipe_dist_to_player"])
	features_list.append(state["player_y"] - state["next_pipe_top_y"])
	features_list.append(state["player_y"] - state["next_pipe_bottom_y"])
	return np.array(features_list).reshape(1,5)

def build_model():
	"""
	Build NN model
	
	Returns
	-------
	object
	    Built NN model
	"""
	model = Sequential()
	model.add(InputLayer((5,)))
	model.add(Dense(50, activation="sigmoid"))
	model.add(Dense(50 , activation="relu"))
	model.add(Dense(2 , activation= "linear"))
	model.compile(keras.optimizers.Adagrad(), loss = "mse", metrics = ["accuracy"])
	return model

def save_model(model):
	"""
	Save function for the model weights
	
	Parameters
	----------
	model : object
	    Trained NN model
	"""
	model.save_weights("weights.h5")

def load_model(model):
	"""
	Load function
	
	Parameters
	----------
	model : object
	    Plain NN model
	
	Returns
	-------
	object
	    Compiled model with weights updated from the save file
	"""
	model.load_weights("weights.h5")
	model.compile(keras.optimizers.Adagrad(), loss = "mse", metrics = ["accuracy"])
	return model

def update_model(model, batch):
	"""
	Q-Learning algorithm
	
	Parameters
	----------
	model : object
	    NN model
	batch : list
	    Training data
	"""
	gamma = 0.95
	for features, next_features, index, reward , game_over in batch:
		y = model.predict(features)
		yy = model.predict(next_features)
		if game_over:
			y[0][index] = reward
		else:
			y[0][index] = reward + gamma * np.max(yy[0])
		model.fit(features, y, batch_size = 32, epochs = 1 , verbose=False)

def train(env, model, game):
	"""
	Training function for the model
	
	Parameters
	----------
	env : object
	    PLE enviorement
	model : object
	    NN model
	game : object
	    Selected game (FlappyBird)
	"""
	global eps
	epoch = 0
	total_reward = 5
	experience = [None] * SIZE
	it, number_of_elements = 0, 0
	
	while total_reward < 300:
		game_over = env.game_over()
		features = get_features(game.getGameState())
		index = get_move(model, features)
		reward = env.act(moves[index])
		total_reward += reward
		
		experience[it] = (features, get_features(game.getGameState()), index, reward , game_over)
		it += 1
		if it == SIZE:
			it = 0
		if number_of_elements < SIZE:
			number_of_elements += 1
		
		if env.game_over():
			print("Training epoch: " , epoch , " Reward :" , total_reward)
			epoch += 1
			total_reward = 5
			if number_of_elements >= 32:
				update_model(model, random.sample(experience[:number_of_elements], 32))
				save_model(model)
				eps = max(eps - 0.001, 0)
			env.reset_game()

def play(env, game):
	"""
	Play function for the model
	
	Parameters
	----------
	env : object
	    PLE enviorement
	game : object
	    Selected game (FlappyBird)
	"""
	total_reward = 0
	model = load_model(build_model())
	while True:
		if env.game_over():
			print(total_reward)
			total_reward = 0
			env.reset_game()
		total_reward += env.act(moves[np.argmax(model.predict(get_features(game.getGameState())))])

if __name__ == "__main__":
	total_reward = 0
	game = FlappyBird()
	env = PLE(game, fps = 30, display_screen = True, force_fps = False)
	env.display_screen = True
	env.force_fps = True
	env.init()

	'''Train'''
	model = build_model()
	train(env, model, game)
	
	'''Play'''
	# play(env, game)
