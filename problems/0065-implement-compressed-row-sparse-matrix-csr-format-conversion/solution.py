import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	arr = np.array(dense_matrix)
	row_idx , col_idx = np.nonzero(arr)
	vals = arr[row_idx , col_idx]
	
	num_rows = arr.shape[0]
	row_count = np.bincount(row_idx , minlength = num_rows)

	row_ptr = np.zeros(num_rows + 1, dtype = np.int32)
	row_ptr[1:] = np.cumsum(row_count)
	return vals.tolist(),col_idx.tolist(), row_ptr.tolist()