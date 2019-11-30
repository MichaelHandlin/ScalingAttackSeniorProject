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

    def get_coefficients(arr):
        I_mm = np.identity(m)
        I_nn = np.identity(n)

        IN_max = 255  # Typical maximum pixel value for most image formats

        D_mpm = Image.fromarray(arr).resize((m, m_prime))  # D_m'*m
        CL = np.array(D_mpm) / IN_max

        # normalize CL
        for i in range(m_prime):
            CL_sum = 0
            for j in range(m):
                CL_sum += CL[i, j]
            CL[i, :] = CL[i, :] / CL_sum

        # Find CR
        D_nnp = Image.fromarray(arr).resize((n_prime, n))  # D_n*n'
        CR = np.array(D_nnp) / IN_max

        for i in range(n):
            CR_sum = 0
            for j in range(n_prime):
                CR_sum += CR[i, j]
            CR[i, :] = CR[i, :] / CR_sum

        return CL, CR

    def get_color_arrays(img):
        img_red, img_blue, img_green = img.split()
        out_red = np.array(img_red)
        out_blue = np.array(img_blue)
        out_green = np.array(img_blue)
        return out_red, out_blue, out_green

    def scale_horizontal(red_arr, blue_arr, green_arr, CR):
        scaled_red = np.dot(red_arr, CR)
        scaled_blue = np.dot(blue_arr, CR)
        scaled_green = np.dot(green_arr, CR)
        return scaled_red, scaled_blue, scaled_green

    def scale_vertical(red_arr, blue_arr, green_arr, CL):
        scaled_red = np.dot(CL[0], red_arr)
        scaled_green = np.dot(CL[1], green_arr)
        scaled_blue = np.dot(CL[2], blue_arr)
        print("Scaled_Blue: " + str(scaled_blue.shape))
        return scaled_red, scaled_green, scaled_blue

    n, m = src_img.size
    n_prime, m_prime = tgt_img.size
    attack_image = ImageChops.add(src_img, tgt_img.resize((m, n)))
    atk_r_arr, atk_g_arr, atk_b_arr = get_color_arrays(attack_image)
    #Get color matrices for each
    src_img.convert('RGB')  # Remove alpha channel
    tgt_img.convert('RGB')  # Remove Alpha Channel

    #  Get manageable
    tgt_red_arr, tgt_blue_arr, tgt_green_arr = get_color_arrays(tgt_img)
    src_red_arr, src_blue_arr, src_green_arr = get_color_arrays(src_img)

    #  Get coefficients per channel
    CL_r, CR_r = get_coefficients(src_red_arr)
    CL_g, CR_g = get_coefficients(src_green_arr)
    CL_b, CR_b = get_coefficients(src_blue_arr)

    s_scaled_r, s_scaled_g, s_scaled_b = scale_vertical(src_red_arr, src_blue_arr, src_green_arr, (CL_r, CL_g, CL_b))
    S_mnp = merge_channels(s_scaled_r, s_scaled_b, s_scaled_g, m, n_prime)

    #  Initialize delta arrays
    delta_v1_red = np.zeros(shape=(m, n_prime))
    delta_v1_blue = np.zeros(shape=(m, n_prime))
    delta_v1_green = np.zeros(shape=(m, n_prime))

    S_mnp_img = Image.fromarray(S_mnp)
    temp_atk = ImageChops.add(S_mnp_img, tgt_img.resize((m, n_prime)))
    temp_atk.save("test.jpg", "JPEG")
    temp_atk_r_arr, temp_atk_b_arr, temp_atk_g_arr = get_color_arrays(temp_atk)

    delta_v1_red = get_vert_perturbation(s_scaled_r, tgt_red_arr, CL_r, temp_atk_r_arr, atk_r_arr)
    delta_v1_green = get_vert_perturbation(s_scaled_g, tgt_green_arr, CL_g, temp_atk_g_arr, atk_g_arr)
    delta_v1_blue = get_vert_perturbation(s_scaled_b, tgt_blue_arr, CL_b, temp_atk_b_arr, atk_b_arr)

    attack_mnp_r = s_scaled_r + delta_v1_red
    attack_mnp_b = s_scaled_b + delta_v1_blue
    attack_mnp_g = s_scaled_g + delta_v1_green

    delta_h1_red = np.zeros(shape=(m, n))
    delta_h1_blue = np.zeros(shape=(m, n))
    delta_h1_green = np.zeros(shape=(m, n))
    for row in range(m):  # For each column
        delta_v1_red[row, :] = get_horz_perturbation(s_scaled_r[row, :], tgt_red_arr[row, :], CR_r, attack_mnp_r, atk_r_arr)
        delta_v1_green[row, :] = get_horz_perturbation(s_scaled_g[row, :], tgt_green_arr[row, :], CR_g, attack_mnp_g, atk_g_arr)
        delta_v1_blue[row, :] = get_horz_perturbation(s_scaled_b[row, :], tgt_blue_arr[row, :], CR_b, attack_mnp_b, atk_b_arr)
    attack_r = src_red_arr + delta_v1_red
    attack_b = src_blue_arr + delta_v1_blue
    attack_g = src_green_arr + delta_v1_green
    attack_img_arr = merge_channels(attack_r, attack_b, attack_g, m, n)
    attack_img = Image.fromarray(attack_img_arr)
    return attack_img


# if columnwise,
#   S is column s[:, col], T is T[:, col], CX is CL.
# if row-wise
#   S is row S[row, :], T is A*[row, :], CX is CR.
def get_vert_perturbation(S, T, CL, delta_1, A, obj='min', IN_max=255, epsilon=.01):
    global m_prime
    delta_out = np.zeros(shape=(m, n_prime))
    delta_temp = CL * delta_1

    for i in range(n_prime):
        delta_out[:, i] = np.dot(np.dot(np.transpose(delta_temp[:, i]), np.identity(m_prime)), delta_temp[:, i])

    #print((np.transpose(delta_temp[:, i]) * np.identity(m_prime) * delta_temp[:, i]).shape)
    #for i in range(n_prime):
    #    j = cvx.Variable(integer=range(m_prime))
    #    constraints = (0 <= A[:, i],
    #                   A[:, i] <= IN_max,
    #                   cvx.norm(np.dot(CL, A)[:, i] - T[:, i]) <= epsilon * IN_max

    #    function = cvx.Minimize(cvx.norm(np.dot(np.dot(np.transpose(delta_temp[:, i]), np.identity(m_prime)), delta_temp[:, i])))
    #    problem = cvx.Problem(function, constraints)
    #    delta_out[:, i] = problem.solve(method='dccp')



    # j = cvx.Variable()
    # Ensure constraints are met
    #delta_1T = np.transpose(delta_1)
    #print("CX Shape:", CL.shape)
    #print("Delta_1 shape: ", delta_1.shape)
    #function = cvx.Minimize(cvx.norm(delta_1))
    # function = cvx.Minimize(np.dot(delta_1T[j, :], np.identity(m_prime)) * delta_1[:, j])
    #constraints = ([0 <= A[:, j],
    #                A[:, j] <= IN_max,
    #                cvx.norm(np.dot(CL, A)[:, j] - T, "inf") <= epsilon * IN_max])
    #problem = cvx.Problem(function, constraints)
    #problem = cvx.Problem(function, [0 <= T, T <= IN_max , cvx.norm(CX*delta_1 - T, "inf") < epsilon *IN_max])
    #problem = cvx.Problem(function, [0 <= j, j < len(S)])
    return delta_out


def get_horz_perturbation(S, T, CR, delta_1, A, obj='min', IN_max=255, epsilon=.01):
    global m_prime
    delta_out = np.zeros(shape=(m, n))
    delta_temp = delta_1 * CR

    for i in range(m_prime):
        delta_out[i, :] = np.dot(np.dot(np.transpose(delta_temp[i, :]), np.identity(n_prime)), delta_temp[i, :])
    return delta_out


def merge_channels(red_arr, blue_arr, green_arr, width, height):
    atk_img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    print(atk_img_arr.shape)
    print(red_arr.shape)
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
