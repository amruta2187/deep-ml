import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    import numpy as np
    X_T = np.transpose(X)
    mul1 = np.dot(X_T, X)
    inv_mul = np.linalg.inv(mul1)
    mul2 = np.dot(X_T, y)
    theta = np.dot(inv_mul , mul2)
	return theta