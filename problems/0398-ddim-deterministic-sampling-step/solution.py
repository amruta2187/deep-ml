import numpy as np

def ddim_sample_step(x_t: np.ndarray, noise_pred: np.ndarray, alpha_bar_t: float, alpha_bar_t_prev: float) -> tuple:
    """
    Perform one deterministic DDIM sampling step.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        noise_pred: Predicted noise from the model, shape (D,)
        alpha_bar_t: Cumulative alpha at timestep t
        alpha_bar_t_prev: Cumulative alpha at timestep t-1
    
    Returns:
        Tuple of (pred_x0, x_t_prev), each a numpy array rounded to 4 decimals
    """
    pred_x0 = (x_t - np.sqrt(1 - alpha_bar_t) * noise_pred) / np.sqrt(alpha_bar_t)

    prev_x_t = np.sqrt(alpha_bar_t_prev) * pred_x0 + np.sqrt(1- alpha_bar_t_prev) * noise_pred 

    x0_out = np.sign(pred_x0) * np.floor(np.abs(pred_x0) * 10000 + 0.5 + 1e-7) / 10000

    xt_out = np.sign(prev_x_t) * np.floor(np.abs(prev_x_t) * 10000 + 0.5 + 1e-7) / 10000

    return x0_out, xt_out