import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    max_z = np.max(scores, axis = -1, keepdims = True)
    exp_shifted = np.exp(scores - max_z)

    return exp_shifted / np.sum(exp_shifted , axis = -1, keepdims = True)