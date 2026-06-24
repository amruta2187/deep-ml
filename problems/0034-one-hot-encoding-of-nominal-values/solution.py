import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col is not None:
		num_classes = n_col
		integer_indices = x
	else:
		unique_val, integer_indices = np.unique(x, return_inverse = True)
		num_classes = len(unique_val)
	one_hot = np.eye(num_classes)[integer_indices]
	return one_hot