import ScalingAttack as sa
from PIL import Image
import cvxpy as cvx
import dccp
x = cvx.Variable(2)
y = cvx.Variable(2)
myprob = cvx.Problem(cvx.Maximize(cvx.norm(x-y,2)), [0<=x, x<=1, 0<=y, y<=1])
print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DCCP:", dccp.is_dccp(myprob))  # true
result = myprob.solve(method='dccp')
print("x =", x.value)
print("y =", y.value)
print("cost value =", result[0])



'''
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

'''