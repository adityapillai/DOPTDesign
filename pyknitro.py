from knitro import *
import numpy as np
from scipy.linalg import cholesky
import time



# s is number of vectors we are choosing
# A is a matrix where rows are vectors
# n is total number of vectors(A.shape[0]) we are considering
def run_knitro(A, s):
    n = A.shape[0]
    def callbackEvalF (kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALFC:
                print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
                return -1
            x = evalRequest.x
            W = A.T @ np.diag(x) @ A
            W = 0.5 * (W + W.T)
            try:
                R_f_kn = cholesky(W, lower=False)
                evalResult.obj = 2 * np.sum(np.log(np.diag(R_f_kn)))
            except np.linalg.LinAlgError:
                evalResult.obj = np.inf
            return 0

    def callbackEvalG (kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x

        # Evaluate gradient of nonlinear objective
        
        W = A.T @ np.diag(x) @ A
        W = 0.5 * (W + W.T)
        try:
            R_f_kn = cholesky(W, lower=False)
            Rinv = np.linalg.inv(R_f_kn)
            Winv = Rinv @ Rinv.T
            df_mat = A @ Winv @ A.T
            dx = np.diag(df_mat)
        except np.linalg.LinAlgError:
            dx = np.zeros_like(x)
        for (i,el) in enumerate(dx):
            evalResult.objGrad[i] = el

        return 0
    try:
        kc = KN_new ()
    except:
        print("Failed to find a valid license.")
        #quit ()

    KN_load_param_file (kc, "knitro.opt")
    KN_add_vars (kc, n)
    KN_set_var_lobnds (kc, xLoBnds = np.zeros(n))
    KN_set_var_upbnds (kc, xUpBnds = KN_INFINITY*np.ones(n))
    KN_set_var_primal_init_values (kc, xInitVals = (s/n)*np.ones(n))

    KN_add_cons(kc, 1)
    KN_set_con_eqbnds(kc, cEqBnds = np.array ([s]))
    for i in range(n):
        KN_add_con_linear_struct (kc, 0, i, 1.0)

    cb = KN_add_eval_callback (kc, evalObj = True, funcCallback = callbackEvalF)
    KN_set_cb_grad (kc, cb, objGradIndexVars = KN_DENSE, gradCallback = callbackEvalG)
    KN_set_obj_goal (kc, KN_OBJGOAL_MAXIMIZE)
    start_time = time.perf_counter()
    nStatus = KN_solve (kc)
    elapsed_time = time.perf_counter() - start_time
    nStatus, objSol, x, lambda_ = KN_get_solution (kc)
    return -objSol, np.array(x),elapsed_time
    

'''
# Example usage:
m = 40
n = int(20*m)
s = int(2*m)
name_inst = "A_" + str(n) + "_" + str(m) + "_" + str(s) + ".npz"
A = np.load(name_inst)
print(f"shape of a is {A.shape}")
print(f"expected shape of A is {(n, m)}")

obj_val, x_opt,elapsed_time = run_knitro(A, s)
print("Optimal Objective Value:", -obj_val)
print("Elapsed time:", elapsed_time)
# x_opt are the "weights"
print("Optimal Solution:", x_opt.shape)
'''
