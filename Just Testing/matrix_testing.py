from PIL import Image
import numpy as np
import sys

# np.set_printoptions(threshold=sys.maxsize)
# uncomment to see full matrix of rgb values


'''
1. Open an image
2. Take each pixel and change to an rgb value
    make a matrix of rgb values using numpy
3. Observe original matrix values
4. Scale image using scaling algorithm
5. Observe new matrix values

6. Find the coefficients of the matrix?
'''

im = Image.open('wolf_icon.jpg')
width, height = im.size


matrix = np.array(im)
print(matrix)
print("shape: ", matrix.shape) # dimensions of matrix
print("size: ", matrix.size) # number of elements in matrix

print("example of a value from matrix: ", matrix[1][0]) #specific value from matrix


# converts the numpy matrix/2darray into a PIL image
matrix_to_img = Image.fromarray(matrix)

# generates the wolf picture
matrix_to_img.show()


