def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
    import numpy as np
    c1 = np.cov(vectors)

	return c1