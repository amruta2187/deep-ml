import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	max_S  = np.max(scores, axis = -1,keepdims=True)
	x_arr = np.log(np.sum(np.exp(scores - max_S), axis = -1, keepdims=True))
	
	return scores - max_S - x_arr