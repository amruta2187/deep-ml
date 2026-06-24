import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):

	mean = np.mean(data, axis = 0, keepdims = True)
	dev = np.std(data, axis = 0, keepdims = True)

	dev[dev == 0] = 1e-8
	standardized_data = (data - mean) / dev 

	#normalizatioon
	x_min  = np.min(data, axis = 0 , keepdims = True)
	x_max = np.max(data, axis = 0, keepdims = True)
	denom = (x_max - x_min)
	denom[denom == 0] = 1e-8
	normalized_data = (data - x_min) / denom
	return standardized_data, normalized_data