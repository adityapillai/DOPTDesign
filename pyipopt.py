import cyipopt
import numpy as np
from scipy.linalg import cholesky
from cyipopt import minimize_ipopt
import numpy as np
import time

def natural_bound_dopt_f(x, A):
    W = A.T @ np.diag(x) @ A
    W = 0.5 * (W + W.T)
    try:
        R_f_kn = cholesky(W, lower=False)
        fval = 2 * np.sum(np.log(np.diag(R_f_kn)))
        return -fval
    except np.linalg.LinAlgError:
        return np.inf

def natural_df(x, A):
    W = A.T @ np.diag(x) @ A
    W = 0.5 * (W + W.T)
    try:
        R_f_kn = cholesky(W, lower=False)
        Rinv = np.linalg.inv(R_f_kn)
        Winv = Rinv @ Rinv.T
        df_mat = A @ Winv @ A.T
        dx = np.diag(df_mat)
        return -dx
    except np.linalg.LinAlgError:
        dx = np.zeros_like(x)
        return -dx

# s is number of vectors we are choosing
# A is a matrix where rows are vectors
# n is total number of vectors(A.shape[0]) we are considering
def run_ipopt(A, s):
    n = A.shape[0]

    def objective(x):
        fval = natural_bound_dopt_f(x, A)
        return fval

    def gradient(x):
        dx = natural_df(x, A)
        return dx

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - s})
    bounds = [(0, None)] * n
    start_time = time.time()
    result = minimize_ipopt(objective, np.ones(n), jac=gradient, constraints=constraints, bounds=bounds, options={'max_iter': 1000, 'tol': 1e-5,'acceptable_tol': 1e-5})
    elapsed_time = time.time() - start_time

    return result.fun, result.x,elapsed_time

'''
# Example usage:
m = 40
n = int(20*m)
s = int(2*m)
name_inst = "A_" + str(n) + "_" + str(m) + "_" + str(s) + ".npz"
A = np.load(name_inst)
print(f"shape of a is {A.shape}")
print(f"expected shape of A is {(n, m)}")

obj_val, x_opt,elapsed_time = run_ipopt(A, s)
print("Optimal Objective Value:", -obj_val)
print("Elapsed time:", elapsed_time)
# x_opt are the "weights"
print("Optimal Solution:", x_opt.shape)
'''
