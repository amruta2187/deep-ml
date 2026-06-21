import numpy as np

def make_diagonal(x):
	# Your code here
	vector = np.array(x)

	mat = np.diag(vector)
	return mat