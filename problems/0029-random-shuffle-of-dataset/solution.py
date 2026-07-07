import numpy as np

def shuffle_data(X, y, seed=None):
	if seed is not None:
		np.random.seed(seed)
	indices = np.arange(len(X))
	shuffled_indices = np.random.permutation(indices)
   
	X_shuffled = X[shuffled_indices]
	y_shuffled = y[shuffled_indices]

	return X_shuffled, y_shuffled