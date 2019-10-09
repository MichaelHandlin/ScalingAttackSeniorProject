import ScalingAttack as sa
from PIL import Image

try:
    src_img = Image.open("images/sheep.jpg")
    tgt_img = Image.open("images/wolf_icon.jpg")
except IOError:
    print("One of the files was not found.")
CR_red = sa.get_coefficients(src_img, tgt_img, "R", "R")
CR_green = sa.get_coefficients(src_img, tgt_img, "G", "R")
CR_blue = sa.get_coefficients(src_img, tgt_img, "B", "R")

CL_red = sa.get_coefficients(src_img, tgt_img, "R", "L")
CL_green = sa.get_coefficients(src_img, tgt_img, "G", "L")
CL_blue = sa.get_coefficients(src_img, tgt_img, "B", "L")

