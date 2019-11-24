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
    tgt_red_arr, tgt_blue_arr, tgt_green_arr = get_color_arrays(tgt_red, tgt_blue, tgt_green)
    src_red_arr, src_blue_arr, src_green_arr = get_color_arrays(src_red, src_blue, src_green)
    s_scaled_r, s_scaled_b, s_scaled_g = scale_vertical(src_red_arr, src_blue_arr, src_green_arr, CL)
    S_mnp = merge_channels(s_scaled_r, s_scaled_b, s_scaled_g, m, n_prime)
    delta_v1_red, delta_v1_blue, delta_v1_green = np.zeros(shape=(m, n_prime))
    #  delta_v1_blue = np.zeros(shape=(m, n_prime))
    #  delta_v1_green = np.zeros(shape=(m, n_prime))
    S_mnp_img = Image.fromarray(S_mnp)
    temp_atk = ImageChops.add(S_mnp_img, tgt_img.resize((m, n_prime)))
    temp_atk_r, temp_atk_b, temp_atk_g = temp_atk.split()
    for col in range(n_prime):  # For each column
        delta_v1_red[:, col] = get_perturbation(s_scaled_r[:, col], tgt_red_arr[:, col], CL, temp_atk_r)
        delta_v1_blue[:, col] = get_perturbation(s_scaled_b[:, col], tgt_blue_arr[:, col], CL, temp_atk_b)
        delta_v1_green[:, col] = get_perturbation(s_scaled_g[:, col], tgt_green_arr[:, col], CL, temp_atk_g)
    attack_mnp_r = s_scaled_r + delta_v1_red
    attack_mnp_b = s_scaled_b + delta_v1_blue
    attack_mnp_g = s_scaled_g + delta_v1_green

    delta_h1_red, delta_h1_blue, delta_h1_green = np.zeros(shape=(m, n))
    for row in range(m):  # For each column
        delta_v1_red[row, :] = get_perturbation(s_scaled_r[row, :], tgt_red_arr[row, :], CR, attack_mnp_r)
        delta_v1_blue[row, :] = get_perturbation(s_scaled_b[row, :], tgt_blue_arr[row, :], CR, attack_mnp_b)
        delta_v1_green[row, :] = get_perturbation(s_scaled_g[row, :], tgt_green_arr[row, :], CR, attack_mnp_g)
    attack_r = src_red_arr + delta_v1_red
    attack_b = src_blue_arr + delta_v1_blue
    attack_g = src_green_arr + delta_v1_green
    attack_img_arr = merge_channels(attack_r, attack_b, attack_g, m, n)
    attack_img = Image.fromarray(attack_img_arr)
    return attack_img


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

    print(CL.shape)
    print(CR.shape)
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

    print(CL.shape)
    print(CR.shape)
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
    problem = cvx.Problem(function, [0 <= T[:, j], T[:, j] <= IN_max, cvx.norm(CX*A[:, j] - T[:, j], "inf") < epsilon *IN_max])
    #problem = cvx.Problem(function, [0 <= j, j < len(S)])
    return problem.solve()


def get_color_arrays(img_red, img_blue, img_green):
    out_red = np.array(img_red)
    out_blue = np.array(img_blue)
    out_green = np.array(img_blue)
    return out_red, out_blue, out_green


def scale_horizontal(red_arr, blue_arr, green_arr, CR):
    print(red_arr.shape)
    scaled_red = np.dot(red_arr, CR)
    scaled_blue = np.dot(blue_arr, CR)
    scaled_green = np.dot(green_arr, CR)
    return scaled_red, scaled_blue, scaled_green


def scale_vertical(red_arr, blue_arr, green_arr, CL):
    scaled_red = np.dot(CL, red_arr)
    scaled_blue = np.dot(CL, blue_arr)
    scaled_green = np.dot(CL, green_arr)
    return scaled_red, scaled_blue, scaled_green


def merge_channels(red_arr, blue_arr, green_arr, width, height):
    atk_img_arr = np.zeros((width, height, 3), dtype=np.uint8)
    for y in range(atk_img_arr.shape[0]):
        for x in range(atk_img_arr.shape[1]):
            atk_img_arr[y][x][0] = red_arr[y][x]
            atk_img_arr[y][x][1] = blue_arr[y][x]
            atk_img_arr[y][x][2] = green_arr[y][x]

    return atk_img_arr

# Terminates the program if an error is detected
def terminate(error):
    print(error)
    exit(1)
