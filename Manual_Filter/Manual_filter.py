import numpy as np

def get_matrix_input(size, name):
    print(f"Enter the values for the {size}x{size} {name} matrix row-wise, separated by spaces:")
    matrix = []
    for i in range(size):
        while True:
            row_input = input(f"Row {i+1}: ")
            row_values = row_input.strip().split()
            if len(row_values) != size:
                print(f"Please enter exactly {size} values.")
            else:
                try:
                    row = [float(val) for val in row_values]
                    matrix.append(row)
                    break
                except ValueError:
                    print("Please enter valid numbers.")
    return np.array(matrix)

matrix_size = 5
matrix = get_matrix_input(matrix_size, "11x11")

filter_size = 3
filter_kernel = get_matrix_input(filter_size, "3x3 filter")

def convolve2d(matrix, filter):
    m_rows, m_cols = matrix.shape
    f_rows, f_cols = filter.shape

    output_rows = m_rows - f_rows + 1
    output_cols = m_cols - f_cols + 1

    output = np.zeros((output_rows, output_cols))

    for i in range(output_rows):
        for j in range(output_cols):
            region = matrix[i:i+f_rows, j:j+f_cols]
            output[i, j] = np.sum(region * filter)
    return output

result = convolve2d(matrix, filter_kernel)

print("\nConvolution Result:")
print(result)