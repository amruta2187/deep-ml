import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	inv_c = np.linalg.inv(C)
	P = np.dot(inv_c , B)
	return P