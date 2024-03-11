import utilZ
import numpy as np
import time
from functools import partial
import logging
logger = logging.getLogger(__name__)

import sys

def general_local_alg(S, current_val, local_move_fun):
    iterations, ip_iter = 0, 0
    logging.info(f"Iteration 0, value {np.log(current_val):.5f}")
    #vals = [current_val]
    #sep_IP = []
    while True:
        new_val, replace_index, new_vec, solved_ip = local_move_fun(S, current_val)
        #vals.append(new_val)
        #sep_IP.append(solved_ip)
        solve = "IP" if solved_ip else "heuristic"
        iterations += 1
        logging.info(f"Iteration {iterations}, value {np.log(new_val):.5f}, used {solve}")
        ip_iter += int(solved_ip)
        if new_val - current_val < 10 ** (-5):
            break
        current_val = new_val
        S[:, replace_index] = new_vec

    return current_val, iterations, ip_iter #, np.array(vals), np.array(sep_IP)


def local_move_pairs(S, old_val, levels, Pairs):
    d, k = S.shape
    # starting point
    pm = utilZ.pairsMat(S, Pairs)

    pmpm = pm @ pm.T

    dS = np.linalg.det(pmpm)
    Sinv = np.linalg.inv(pmpm)

    denominators = 1 - (pm.T.dot(Sinv) * pm.T).sum(axis=1)
    replace_index, addVector, maxVal = 0, 0, 0
    ip_inputs = []
    for i in range(k):
        c = pm[:, i].reshape(-1, 1)
        tmp = c @ c.T

        Ri = pmpm - tmp
        dRi = np.linalg.det(Ri) if denominators[i] > 10 ** (-5) else 0

        c = c.ravel()
        QM = Sinv + (Sinv @ tmp @ Sinv) / denominators[i] if dRi else -1 * np.linalg.pinv(Ri, hermitian=True) @ Ri

        ls_val, vec, t = utilZ.localSearch(QM, Pairs, levels=levels)
        ip_inputs.append((QM, dRi, vec))

        currentVal = dRi * (1 + ls_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (vec @ vec + ls_val)
        #
        # currentVal = dRi*(1 + m.getObjective().getValue())
        if currentVal > maxVal:
            replace_index = i
            addVector = vec
            maxVal = currentVal

    if maxVal > old_val or levels:
        return maxVal, replace_index, addVector, False

    for i in range(k):
        QM, dRi, vec = ip_inputs[i]
        c = pm[:, i]

        ip_val, vec, t = utilZ.quadIP(QM, pairs=Pairs, start=vec) if not levels else utilZ.quadIP_two(QM, Pairs,
                                                                                                      start=vec)
        currentVal = dRi * (1 + ip_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (vec @ vec + ip_val)

        if currentVal > maxVal:
            replace_index = i
            addVector = vec
            maxVal = currentVal
        if currentVal > old_val: break

    return maxVal, replace_index, addVector, True


def local_alg_pairs(k, d, pairs, levels=False):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}
    trials = 50
    arr_s = np.ones((trials, d, k))

    if not levels:
        utilZ.random_b(arr_s)
    else:
        utilZ.random_l(arr_s)
    arr_s_pairs = np.array([utilZ.pairsMat(S, pairs) for S in arr_s])
    current_val, max_idx = utilZ.best_starting_sol(arr_s_pairs)
    S = arr_s[max_idx]

    if current_val < 10 ** (-5):
        logging.error(f"Could not find non-zero starting solution among {trials} random solutions.")
        return None

    fun = partial(local_move_pairs, levels=levels, Pairs=pairs)

    current_val, iterations, ip_iter = general_local_alg(S, current_val, fun)

    info["Final Value"] = np.log(current_val)
    info["IP Iterations"] = ip_iter
    info["Iterations"] = iterations
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    return info


def local_move(S, prev_value, ones_bound):
    d, k = S.shape
    SS = S @ S.T
    dS = np.linalg.det(SS)
    S_inv = np.linalg.inv(SS)

    denominators = 1 - (S.T.dot(S_inv) * S.T).sum(axis=1)
    replace_index, add_vector, max_val = 0, 0, 0
    ip_inputs = []
    for i in range(k):
        c = S[:, i].reshape(-1, 1)
        tmp = c @ c.T

        Ri = SS - tmp
        dRi = np.linalg.det(Ri) if denominators[i] > 10 ** (-3) else 0

        c = c.ravel()
        QM = S_inv + (S_inv @ tmp @ S_inv) / denominators[i] if dRi else -np.linalg.pinv(Ri, hermitian=True) @ Ri

        if not dRi:
            ip_inputs.append((QM, dRi, None))
            continue

        # use local first if all of them fail then solve IP
        sol_val, vec, t = utilZ.localSearch(QM, ones_bound=ones_bound)
        ip_inputs.append((QM, dRi, vec))

        current_val = dRi * (1 + sol_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (vec @ vec + sol_val)

        if current_val > max_val:
            replace_index = i
            add_vector = vec
            max_val = current_val

    if max_val > prev_value:
        return max_val, replace_index, add_vector, False

    for i in range(k):
        QM, dRi, vec = ip_inputs[i]
        sol_val, vec, t = utilZ.quadIP(QM, start=vec, onesBound=ones_bound)

        c = S[:, i]
        current_val = dRi * (1 + sol_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (vec @ vec + sol_val)

        if current_val > max_val:
            replace_index = i
            add_vector = vec
            max_val = current_val

    return max_val, replace_index, add_vector, True



def update_inverse_rank2(SS_inv,r,wZ,wZw):
    Y  = SS_inv - (np.outer(wZ.T,wZ))/(1+wZw)
    rY = r.T@Y
    SS_inv_new = Y + (np.outer(rY.T,rY))/(1-np.dot(rY,r))
    return SS_inv_new



def update_logdet_rank2(SS_inv,r,Zr,idx,diff,obj_val):
    w = r.copy()
    w[idx] += diff
    wZ = w.T@SS_inv
    wZw = wZ@w
    rZr = r.T@Zr
    wZr = w.T@Zr
    exp_new_val = (1+wZw)*(1 - rZr) + wZr**2
    if exp_new_val >= 1:
        return obj_val + np.log(exp_new_val),wZ,wZw
    else:
        return -np.inf,0,0
    
    
class Results:
    def __init__(self, ip_iter=0,ls_iter=0,swaps=0,ls_loop=[0,0,0]):
        self.ip_iter = ip_iter
        self.ls_iter = ls_iter
        self.swaps = swaps
        self.ls_loop = ls_loop


def changeBit(S, SS_inv, nrows,ncols, ldet_S,ones_bound=None):
    d, k = nrows,ncols
    if ones_bound is None:
        ones_bound = d
    flag,flagbit = True,False
    res = Results()
    while flag:
        flag = False
        for i in range(k):
            r = S[:,i].copy()
            Zr = SS_inv@r
            for j in range(1,d):
                sji = S[j,i]
                if sji == 1 or (sji == 0 and np.sum(S[:,i]) < ones_bound):
                    if sji == 1:
                        my_diff = -1
                    else:
                        my_diff = 1
                    new_ldet,wZ,wZw = update_logdet_rank2(SS_inv,r,Zr,j,my_diff,ldet_S)
                    if new_ldet > ldet_S:
                        SS_inv = update_inverse_rank2(SS_inv,r,wZ,wZw)
                        ldet_S = new_ldet
                        S[j,i] += my_diff
                        flag = True
                        flagbit = True
                        res.swaps += 1
                        break
            if flag:
                break
    return S,SS_inv,ldet_S,res,flagbit
    


def reduceDimension(arr_s,trials,k,ones_bound=None):
    res = Results()
    new_arr_s,ldet_arr_s = [],[]
    for j in range(trials):    
        S = arr_s[j]
        d,nmax = S.shape
        SS = S @ S.T
        SS_inv = np.linalg.inv(SS)
        ldet_S = np.linalg.slogdet(SS)[1]
        while True:
            S,SS_inv,ldet_S,res_iter,flagbit = changeBit(S, SS_inv, d, nmax, ldet_S,ones_bound)
            res.swaps += res_iter.swaps 
            best_cZc,best_Zc = 0,0
            max_ldet,idx_max = -np.inf,-np.inf
            if nmax == k:
                break
            for i in range(nmax):
                c = S[:, i].reshape(-1, 1)
                Zc = SS_inv@c
                cZc = c.T@Zc
                new_ldet = np.log(1-cZc) + ldet_S
                if new_ldet > max_ldet:
                    best_cZc,best_Zc = cZc,Zc
                    max_ldet,idx_max = new_ldet,i            
            SS_inv = SS_inv + (1/(1 - best_cZc))*np.outer(best_Zc.T,best_Zc)
            S = np.delete(S, idx_max, axis=1)
            ldet_S = max_ldet
            nmax -= 1

        check_ldet = np.linalg.slogdet(S@S.T)[1]
        print(check_ldet-ldet_S) 
        new_arr_s.append(S)
        ldet_arr_s.append(ldet_S)
    return new_arr_s,ldet_arr_s,res



def general_local_alg_one_iter(S, SS_inv, ldet_S, local_move_fun):
    flag, ip_iter = False,0
    logging.info(f"Iteration 0, value {ldet_S[0,0]:.5f}")
    new_val, replace_index, new_vec, solved_ip = local_move_fun(S, np.exp(ldet_S))
    new_val = np.log(new_val)
    solve = "IP" if solved_ip else "heuristic"
    logging.info(f"Iteration {0}, value {new_val:.5f}, used {solve}")
    r = S[:,replace_index].copy()
    w = new_vec
    Zr = SS_inv@r
    wZ = w.T@SS_inv
    wZw = wZ@w
    rZr = r.T@Zr
    wZr = w.T@Zr
    exp_new_val = (1+wZw)*(1 - rZr) + wZr**2
    if exp_new_val >= 1:
        flag = True
        new_ldet = ldet_S + np.log(exp_new_val)
        SS_inv = update_inverse_rank2(SS_inv,r,wZ,wZw)
        S[:, replace_index] = new_vec
    else:
        flag = False
        new_ldet = ldet_S
    
    return S,SS_inv,new_ldet, ip_iter,flag

    
def doNewLocalSearch(arr_s,ldet_arr_s,res,ones_bound=None):
    S_best,ldet_S_best,ldet_best_init = -np.inf,-np.inf,-np.inf
    local_move_fun = partial(local_move, ones_bound=ones_bound)
    for idx,S in enumerate(arr_s):
        d,k = S.shape
        ldet_S = ldet_arr_s[idx]
        SS_inv = np.linalg.inv(S @ S.T)
        while True:
            res.ls_loop[2] += 1
            ldet_Sold = ldet_S
            S,SS_inv,ldet_S, ip_iter_inner,flag = general_local_alg_one_iter(S, SS_inv, ldet_S, local_move_fun)
            print("===========")
            print(ldet_S-ldet_Sold)
            print(np.linalg.slogdet(S@S.T)[1] - ldet_S)
            print(np.linalg.norm(SS_inv - np.linalg.inv(S@S.T)))
            res.ip_iter += ip_iter_inner
            res.ls_iter += 1
            if flag:
                res.ls_loop[0] += 1
            else:
                break
            S,SS_inv,ldet_S,res_iter,flagbit = changeBit(S, SS_inv, d, k, ldet_S,ones_bound)
            res.swaps += res_iter.swaps
            if res_iter.swaps > 0:
                print("swaps")
                print(res_iter.swaps) 
            if flagbit:
                res.ls_loop[1] += 1
        if ldet_S > ldet_S_best:
            ldet_S_best = ldet_S
            S_best = S.copy
            ldet_best_init = ldet_arr_s[idx]

    ldet_S_best,ldet_best_init = ldet_S_best[0,0],ldet_best_init[0,0]

    return S_best,ldet_S_best,ldet_best_init,res
        


def local_alg(k, d, ones_bound=None):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}

    #previous method
    trials = 50
    arr_s = np.ones((trials, d, k))
    if ones_bound is None:
        utilZ.random_b(arr_s)
    else:
        for S in arr_s: utilZ.random_p(S, ones_bound)
        info["Ones Bound"] = ones_bound
    current_val, max_idx = utilZ.best_starting_sol(arr_s)
    Sold = arr_s[max_idx]
    ldet_init_old = np.linalg.slogdet(Sold@Sold.T)[1]
    
    # new method
    trials = 2
    arr_s = np.ones((trials, d, 2*k))

    if ones_bound is None:
        utilZ.random_b(arr_s)
    else:
        for S in arr_s: utilZ.random_p(S, ones_bound)
        info["Ones Bound"] = ones_bound

    

    arr_s,ldet_arr_s,res = reduceDimension(arr_s,trials,k)
    S,ldet_S,ldet_init,res = doNewLocalSearch(arr_s,ldet_arr_s,res)
    print(S.shape)
    print(ldet_S)
    print(res.__dict__)
   

    


    # could not find a non-zero starting solution
    # if current_val < 10 ** (-3):
    #     logging.error(f"Could not find a non-zero starting solution among {trials} random solutions.")
    #     return None

    # fun = partial(local_move, ones_bound=ones_bound)
    # current_val, iterations, ip_iter = general_local_alg(S, current_val, fun)

    #, vals, sep_ip

    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["IP Iterations"] = res.ip_iter
    info["Iterations"] = res.ls_iter
    info["Bit swaps"] = res.swaps
    info["LS loop"] = res.ls_loop
    info["init_objval_old"] = ldet_init_old
    info["init_objval"] = ldet_init
    info["Final Value"] = ldet_S
    return info #, vals, sep_ip




if __name__ == "__main__":
    d = 28
    k = 2 * d
    q = 5
    P = [(1, 3), (4, 5), (6, 14), (8, 17)]

    # 128.79943
    logging.basicConfig(format='[%(levelname)-2s %(filename)s:%(lineno)d] %(message)s',
                        #datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)



    info = local_alg(k, d, d // 3)

    # plt.plot( )

    #print(local_alg_ob(k, d, q))

    #print(local_alg_pairs(28 * 2, 28, P))
    # print(local_alg(52, 26, ones_bound=26 // 3))
