import cvxpy as cvx
import dccp

x = cvx.Variable(2)
problem = cvx.Problem(cvx.max, [0 <= x, x <= 2])