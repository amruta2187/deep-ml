import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	X = np.array(features)
	y = np.array(labels)
	w = np.array(weights)

	z = np.dot(X,w) + bias 
	probabilities = 1/ (1 + np.exp(-z))

	mse = np.mean((probabilities - y)**2)
	round_prob = np.round(probabilities, 4).tolist()
	round_mse = round(float(mse), 4)
	return round_prob, round_mse