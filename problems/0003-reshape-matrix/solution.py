import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    original_size = len(a) * len(a[0]) if a else 0
    new_rows,new_cols = new_shape

    if original_size != new_rows * new_cols:
        return []
        
    import numpy as np
    arr = np.array(a)
    # check if can be reshaped
    
    new_arr = arr.reshape(new_shape)
	return new_arr