import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	batches = []
	for i in range(0, len(X), batch_size):
		batch_x = X[i : i + batch_size]
		if y is not None:
			batch_y = y[i: i + batch_size]
			batches.append([batch_x.tolist(), batch_y.tolist()])
		else:
            # Convert X to a native list
			batches.append(batch_x.tolist())
	return batches