import numpy as np

def accuracy_score(y_true, y_pred):
	accurate = np.count_nonzero(y_true == y_pred)
	missed = np.count_nonzero(y_true != y_pred)
	
	return accurate/len(y_pred) 
