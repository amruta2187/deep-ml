
import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v_arr = np.array(v)
	L_arr = np.array(L)

	num = np.dot(v_arr,L_arr)
	denom = np.dot(L_arr,L_arr)

	proj = (num / denom) * L_arr

	return proj.tolist()	
