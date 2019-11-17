from PIL import Image
from PIL import ImageChops
import numpy as np
import cvxpy as cvx
import dccp

tgt_img = Image.Image
src_img = Image.Image
n = 0
m = 0
n_prime = 0
m_prime = 0

def create_attack_image(src_img, tgt_img, scale_func):
    global m
    global n
    global m_prime
    global n_prime

    m, n = src_img.size
    m_prime, n_prime = tgt_img.size
    CL, CR = get_coefficients(m, n, m_prime, n_prime)
    #Get color matrices for each
    src_img.convert('RGB')  # Remove alpha channel
    tgt_img.convert('RGB')  # Remove Alpha Channel
    tgt_red, tgt_green, tgt_blue = tgt_img.split()
    src_red, src_green, src_blue = src_img.split()
    tgt_red_arr = np.array(tgt_red)
    tgt_blue_arr = np.array(tgt_blue)
    tgt_green_arr = np.array(tgt_green)
    tgt_arrs = [tgt_red_arr, tgt_blue_arr, tgt_green_arr]
    src_red_arr = np.array(src_red)
    src_red_scaled = np.dot(np.dot(CL, src_red_arr), CR)
    src_blue_arr = np.array(src_blue)
    src_blue_scaled = np.dot(np.dot(CL, src_blue_arr), CR)
    src_green_arr = np.array(src_green)
    src_green_scaled = np.dot(np.dot(CL, src_green_arr), CR)
    src_arrs = [src_red_arr, src_blue_arr, src_green_arr]
    #(Image.fromarray(np.dot(np.dot(CL, src_green_arr), CR)).convert("RGB")).save("test.jpg")
    #Successfully makes a scaled image.
    atk_img_arr = np.zeros((n, m, 3), dtype=np.uint8)
    CL_inv, CR_inv = _get_inv_coeff(m, n, m_prime, n_prime)
    tgt_red_scaled = np.dot(np.dot(CL_inv, tgt_red_arr), CR_inv)
    (Image.fromarray(src_red_scaled).convert("RGB")).save("tgt_red_scaled.png")
    tgt_blue_scaled = np.dot(np.dot(CL_inv, tgt_blue_arr), CR_inv)
    tgt_green_scaled = np.dot(np.dot(CL_inv, tgt_green_arr), CR_inv)
    for y in range(atk_img_arr.shape[0]):
        for x in range(atk_img_arr.shape[1]):
            atk_img_arr[y][x][0] = tgt_red_scaled[y][x]
            atk_img_arr[y][x][1] = tgt_green_scaled[y][x]
            atk_img_arr[y][x][2] = tgt_blue_scaled[y][x]
    test_image = ImageChops.add(src_img, Image.fromarray(atk_img_arr))
    test_image = test_image.resize((n_prime, m_prime))
    test_image.save("test1.png")
    #ImageChops.add(Image.fromarray(atk_img_arr), tgt_img).save("test.jpg")
    atk_arrs = []
    #for i in range(3):
    #    atk_arrs[i] = Image.fromarray()


    #tgt_blue_img = Image.fromarray(tgt_blue)
    # Create temp image to initialize delta v1
    #temp_img = Image.fromarray(CL * np.array(tgt_img) * CR) #operands could not be broadcast together with shapes (900,256) (256,256,3)
    # are those color channels?
    # Missing a step here to get the shapes to be compatible.
    # np.array(Image.open(src), np.float)
    # temp_img.show()
    #temp_tgt_img = ImageChops.add(src_img, Image.fromarray(CL * np.array(tgt_img) * CR))
    #temp_tgt_img.show()
    #temp_delta_v1 = np.array(temp_tgt_img) - np.array(src_img)
    #delta_v1 = np.zeros(shape=(m, n_prime))
    #S_mnp = np.array(src_img.resize((m, n_prime), scale_func))
    #for col in range(n_prime):
    #    delta_v1[:, col] = get_perturbation(S_mnp[:, col], tgt_img[:, col], temp_delta_v1, CL)
    #A_mnp = S_mnp + delta_v1

    #delta_h1 = np.zeros(shape=(m, n))
    #for row in range(m):
    #    delta_h1[row, :] = get_perturbation(src_img[row, :], A_mnp[row, :], CR)
    #A_mn = src_img + delta_h1
    #return Image.fromarray(A_mn)
    return Image.fromarray(CL)


def get_coefficients(m, n, m_prime, n_prime):
    I_mm = np.identity(m)
    I_nn = np.identity(n)

    IN_max = 255  # Typical maximum pixel value for most image formats
    D_mpm = Image.fromarray(np.array(I_nn * IN_max)).resize((n, n_prime))  # D_m'*m
    CL = np.array(D_mpm) / IN_max

    # Find CR
    S_arr = np.array(I_nn)
    D_nnp = Image.fromarray(np.array(I_mm * IN_max)).resize((m_prime, m))  # D_n*n'
    CR = np.array(D_nnp) / IN_max

    return CL, CR

def _get_inv_coeff(m, n, m_prime, n_prime):
    I_mm = np.identity(m_prime)
    I_nn = np.identity(n_prime)

    IN_max = 255  # Typical maximum pixel value for most image formats
    D_mpm = Image.fromarray(np.array(I_mm * IN_max)).resize((n_prime, n))  # D_m'*m
    CL = np.array(D_mpm) / IN_max

    # Find CR
    S_arr = np.array(I_nn)
    D_nnp = Image.fromarray(np.array(I_nn * IN_max)).resize((m, m_prime))  # D_n*n'
    CR = np.array(D_nnp) / IN_max

    return CL, CR

# if columnwise,
#   S is column s[:, col], T is T[:, col], CX is CL.
# if row-wise
#   S is row S[row, :], T is A*[row, :], CX is CR.
def get_perturbation(S, T, CX, delta_1, obj='min', IN_max=255, epsilon=.001):
    global m_prime
    # Ensure constraints are met
    j = cvx.Variable(2)
    delta_1T = np.transpose(delta_1)
    function = cvx.Minimize(delta_1T[:, j] * np.identity(m_prime) * delta_1[:, j])
    problem = cvx.Problem(function, [0 <= T[:, j], T[:, j] <= IN_max, cvx.norm(CX*A[:, j] - T[:, j], "inf")] < epsilon *IN_max)
    #problem = cvx.Problem(function, [0 <= j, j < len(S)])
    return problem.solve()


# Terminates the program if an error is detected
def terminate(error):
    print(error)
    exit(1)


    '''
def get_coefficients(src_img, tgt_img, color, side):
    print("get_coefficients called.")
    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size
    print(str(src_w) + " " + str(src_h))
    src_g = list(src_img.getdata(1))That's what
    src_b = list(src_img.getdata(2))

    if color == "R":
        src_r = list(src_img.getdata(0))
        src_np_r = np.array(src_r)
        src_np_r_2d = np.reshape(src_np_r, (src_w, src_h))

        if side == "R":
            CR = _get_right(src_np_r_2d, src_w, tgt_h)
        elif side == "L":
            CL = _get_left(src_np_r_2d, tgt_w, src_h)
    elif color == "G":
        src_g = list(src_img.getdata(1))
        src_np_g = np.array(src_g)
        src_np_g_2d = np.reshape(src_np_g, (src_w, src_h))
        if side == "R":
            CR = _get_right(src_np_g_2d, src_w, tgt_h)
        elif side == "L":
            CL = _get_left(src_np_g_2d, tgt_w, src_h)
    elif color == "B":
        src_b = list(src_img.getdata(2))
        src_np_b = np.array(src_b)
        src_np_b_2d = np.reshape(src_np_b, (src_w, src_h))
        if side == "R":
            print("BR todo")
        elif side == "L":
            print("BL todo")
    return 1


def _get_right(np_arr, src_width, tgt_height):
    q_prime = np.resize(np_arr,(src_width, tgt_height))
    q_inv = np.linalg.pinv(np_arr)
    CR = np.dot(q_inv, q_prime)
    return CR


def _get_left(np_arr, tgt_width, src_height):
    q_prime = np.resize(np_arr, (tgt_width, src_height))  #CR is effectively an identity matrix
    q_inv = np.linalg.pinv(np_arr)
    CL = np.dot(q_prime, q_inv)
    return CL
'''
