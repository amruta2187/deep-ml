
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	y_arr = np.array(y)
	total = y_arr.size
	if total == 0:
		return 0.0
	_ , count = np.unique(y_arr, return_counts = True)

	proba = count / total

	val = 1 - np.sum(proba ** 2)

	return val