from PIL import Image
import numpy as np

def get_coefficients(src_img, tgt_img, color, side):
    print("get_coefficients called.")
    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size
    print(str(src_w) + " " + str(src_h))
    src_g = list(src_img.getdata(1))
    src_b = list(src_img.getdata(2))

    if color == "R":
        src_r = list(src_img.getdata(0))
        src_np_r = np.array(src_r)
        src_np_r_2d = np.reshape(src_np_r, (src_w, src_h))

        if side == "R":
            CR = _get_right(src_np_r_2d, src_w, tgt_h)
        elif side == "L":
            print("todo")
            #CL = _get_left(src_np_r_2d, tgt_w, src_h)
    return 1


def _get_right(np_arr, src_width, tgt_height):
    q_prime = np.resize(np_arr,(src_width, tgt_height))
    q_inv = np.linalg.pinv(np_arr)
    CR = np.dot(q_inv, q_prime)
    return CR
