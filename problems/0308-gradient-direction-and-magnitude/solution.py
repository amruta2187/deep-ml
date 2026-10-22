import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	import math
	result = []
	mag = math.sqrt(sum(g ** 2 for g in gradient))

	if mag == 0.0:
		dirctn = [0.0 for g in gradient]
		dec_dirct = [0.0 for g in gradient]
	else:
		dirctn = [g/mag for g in gradient]
		dec_dirct = [-v for v in dirctn]
	return {'magnitude' : mag, 'direction' : dirctn , 'descent_direction':dec_dirct}