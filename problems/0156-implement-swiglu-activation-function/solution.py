import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[-1] //2
    x_v = x[..., :d]
    x_w = x[..., d:]

    sigmoid_x_w = 1/ (1 + np.exp(-(x_w)))
    swish_x_v = x_w * sigmoid_x_w

    scores = swish_x_v * x_v
    return scores