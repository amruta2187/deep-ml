import numpy as np

def batchnorm1d_backward_fused(x: np.ndarray, bngain: np.ndarray, dhpreact: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the fused backward pass through a BatchNorm1d layer.

    Args:
        x:        pre-BN input, shape (N, D)
        bngain:   scale parameter, shape (1, D)
        dhpreact: upstream gradient, shape (N, D)
        eps:      numerical stability constant

    Returns:
        dhprebn: gradient w.r.t. x, shape (N, D)
    """
    N= x.shape[0]
    # Compute mean and variance across the batch dimension
    mean = np.mean(x, axis=0, keepdims=True)
    # ddof=1 applies Bessel's correction (uses N-1 in the denominator)
    var = np.var(x, axis=0, ddof=1, keepdims=True) 
    
    # Compute standard deviation inverse and normalized input (x-hat)
    sigma_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * sigma_inv
    
    # Compute reduction components (Sum over batch axis, i.e., axis 0)
    sum_dhpreact = np.sum(dhpreact, axis=0, keepdims=True)
    sum_dhpreact_x_hat = np.sum(dhpreact * x_hat, axis=0, keepdims=True)
    
    # Analytical fused backward expression
    term1 = N * dhpreact
    term2 = sum_dhpreact
    term3 = (N / (N - 1)) * x_hat * sum_dhpreact_x_hat
    
    dhprebn = bngain * sigma_inv / N * (term1 - term2 - term3)
    return  dhprebn
