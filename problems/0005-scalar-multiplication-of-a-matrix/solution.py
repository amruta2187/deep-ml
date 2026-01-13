def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    import numpy as np
    arr = np.array(matrix)

    result = scalar * arr
	return result