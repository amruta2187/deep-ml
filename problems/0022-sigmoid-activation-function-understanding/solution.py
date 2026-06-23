import math
import numpy as np
def sigmoid(z: float) -> float:
	#Your code here
	denom = (1 + np.exp(-z))
	result = 1/ denom
	return result