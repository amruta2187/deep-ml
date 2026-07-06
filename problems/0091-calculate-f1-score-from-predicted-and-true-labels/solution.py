import numpy as np
def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	TP = np.sum((y_true == 1 ) & (y_pred == 1))
	FP = np.sum((y_true == 0) & (y_pred == 1))
	FN = np.sum((y_true == 1) & (y_pred == 0))

	precision = TP / (TP + FP) if (TP + FP ) > 0.0 else 0.0
	recall = TP / (TP + FN) if (TP + FN ) > 0.0 else 0.0

	if (precision + recall) == 0:
		f1 = 0.0
	else:
		f1 = 2 * ((precision * recall) / (precision + recall))
	return round(f1,3)