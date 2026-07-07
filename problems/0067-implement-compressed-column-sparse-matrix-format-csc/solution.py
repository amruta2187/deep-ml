import numpy as np
def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	matrix = np.array(dense_matrix)
    num_cols = matrix.shape[1]

    col_idx, row_idx = np.nonzero(matrix.T)

    vals = matrix[row_idx, col_idx]

    col_count = np.bincount(col_idx, minlength=num_cols)

    col_ptr = np.zeros(num_cols + 1, dtype=np.int32)
    col_ptr[1:] = np.cumsum(col_count)

    return vals.tolist(), row_idx.tolist(), col_ptr.tolist()
