import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	input_padded = np.pad(input_matrix, pad_width = padding ,mode ='constant', constant_values= 0 )

	output_height = (input_height - kernel_height + 2 * padding)// stride + 1

	output_width = (input_width - kernel_width + 2 * padding) //stride + 1

	output_matrix = np.zeros((output_height , output_width))

	for i in range(output_height):
		for j in range(output_width):
			start_h = i * stride
			end_h = start_h + kernel_height
			start_w =  j * stride
			end_w = start_w + kernel_width

			image_slice = input_padded[start_h:end_h, start_w:end_w]

			output_matrix[i, j] = np.sum(image_slice * kernel)
	return output_matrix
