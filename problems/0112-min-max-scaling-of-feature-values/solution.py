import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    x_min = np.min(x ,axis = -1 , keepdims = True)
    x_max = np.max(x , axis = -1 , keepdims = True)

    numerator = x - x_min
    denom = x_max - x_min
    return numerator / denom
    