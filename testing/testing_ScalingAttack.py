import ScalingAttack2 as sa
import numpy as np
from PIL import Image, ImageChops
import cvxpy as cvx
import dccp
import images
# This functions will test out the functions defined in ScalingAttack.py


def get_perturbation2(source_img):
    source_img.convert('RGB')
    source_img_matrix = np.array(source_img)

    # Take it apart
    def get_R():
        element_list = [element[0] for row in source_img_matrix for element in row]
        R = np.array(element_list)
        R.shape = (len(source_img_matrix[0]), len(source_img_matrix))
        return R

    def get_G():
        element_list = [element[1] for row in source_img_matrix for element in row]
        G = np.array(element_list)
        G.shape = (len(source_img_matrix[0]), len(source_img_matrix))
        return G


    def get_B():
        element_list = [element[2] for row in source_img_matrix for element in row]
        B = np.array(element_list)
        B.shape = (len(source_img_matrix[0]), len(source_img_matrix))
        return B

    # This is where we call the get_perturbation method
    def perturbation_R(R):
        pass
    def perturbation_G(G):
        pass
    def perturbation_B(B):
        pass

    # Now put it back together - the new one after perturbation calculations
    def build_image(R, G, B):
        # From R, take each element and place it at [][][]
        # its pretty, but not mine, find another way? or understand it
        assert R.ndim == 2 and G.ndim == 2 and B.ndim == 2
        rgb = (R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis])
        return Image.fromarray(np.concatenate(rgb, axis=-1))

    def testing():
        element_list = [0 if element <= 0 else element-10 for row in get_R() for element in row]
        matrix = np.array(element_list, dtype='uint8')
        matrix.shape = (len(get_R()[0]), len(get_R()))
        return matrix



    # constraints:
    # 0 <= A[:,j]m*1<=IN_max
    IN_max = 255
    max_constraint1 = IN_max
    # ||CL * A[:,j]m*1-T[:,j]m_prime*1||inf <= ellipson * IN_max
    # max_constraint2 = CL * A
    # Do the perturbation math
    # put the matrices back together and return as image

    return get_R(), testing()




# Test get coefficients

# 7x7 image
im1 = Image.open('C:/Users/selen/PyCharmProjects/ScalingAttackSeniorProject/images/source_test.png')
matrix1 = np.array(im1)
im1_height = len(matrix1)
im1_width = len(matrix1[0])


# print(matrix1)

im2 = im1.resize((2,2), Image.BILINEAR)

matrix2 = np.array(im2)
im2_height = len(matrix2)
im2_width = len(matrix2[0])
# print(matrix2)

matrix3 = np.array([[1, 2, 3], [4, 5, 6]])
CL, CR =sa.get_coefficients(im1_height, im1_width, im2_height, im2_width) # (7,7,2,2)
# print('simple 2x3 matrix:\n',matrix3)

# 7 x 7
print('number of rows/height in original image:', len(matrix1))
print('number of columns/width:',len(matrix1[0]))

print('printing the CL matrix of im1 to im2\n', CL)
# The getcoefficeints will receive the size of source and target and return the coefficient matrices
# and generate an intermediate source image?
print('printing the CR matrix of im1 to im2\n', CR)

print('---------------------------------------------------------------')


x = cvx.Variable(2)
y = cvx.Variable(2)
myprob = cvx.Problem(cvx.Maximize(cvx.norm(x-y,2)), [0<=x, x<=1, 0<=y, y<=1])
print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DCCP:", dccp.is_dccp(myprob))  # true
result = myprob.solve(method = 'dccp')
print("x =", x.value)
print("y =", y.value)
print("cost value =", result[0])


'''Input: scaling function ScaleFunc(), source image Sm⇤n, tar-
get image Tm0⇤n0 , source image size (widths,heights), tar-
get image size (widtht,heightt)'''

print('----------------------MORE TESTING-----------------------------------------')

print(get_perturbation2(im1))

