import utilZ
import numpy as np
import time
from functools import partial
import logging

logger = logging.getLogger(__name__)


def general_local_alg(S, current_val, local_move_fun):
    iterations, ip_iter = 0, 0
    logging.info(f"Iteration 0, value {current_val :.5f}")
    # vals = [current_val]
    # sep_IP = []
    while True:
        #unique_vecs = np.unique(S, axis=1).shape[1]
        #print(f"{unique_vecs=}")
        new_ratio, replace_index, new_vec, solved_ip = local_move_fun(S)
        iterations += 1
        ip_iter += int(solved_ip)

        solve = "IP" if solved_ip else "heuristic"
        logging.info(f"Iteration {iterations}, value {(current_val + np.log(new_ratio)):.5f}, used {solve}")

        if solved_ip and new_ratio < 1 + 10 ** (-3):
            break

        S[:, replace_index] = new_vec
        current_val += np.log(new_ratio)

    return S, current_val, iterations, ip_iter


# todo update pairs with new ls
def local_move_pairs(S, old_val, levels, Pairs):
    d, k = S.shape
    # starting point
    pm = utilZ.pairs_mat(S, Pairs)

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
    arr_s_pairs = np.array([utilZ.pairs_mat(S, pairs) for S in arr_s])
    current_val, max_idx = utilZ.best_starting_sol(arr_s_pairs)
    S = arr_s[max_idx]

    if current_val < 10 ** (-5):
        logging.error(f"Could not find non-zero starting solution among {trials} random solutions.")
        return None

    fun = partial(local_move_pairs, levels=levels, Pairs=pairs)

    S, current_val, iterations, ip_iter = general_local_alg(S, current_val, fun)

    info["Final Value"] = np.log(current_val)
    info["IP Iterations"] = ip_iter
    info["Iterations"] = iterations
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    return info


def local_move(S, ones_bound, A=None, b=None):
    d, k = S.shape
    SS = S @ S.T
    # print(f"vals are {prev_value} {np.linalg.slogdet(S @ S.T)[1]}")
    S_inv = np.linalg.inv(S @ S.T)

    multiples = 1 - (S.T.dot(S_inv) * S.T).sum(axis=1)
    replace_index, add_vector, max_val = 0, 0, 0
    ip_inputs = []

    best_ratio = 0
    all_tmp = np.einsum('ij,ik->ijk', S.T, S.T)
    all_Ri = SS - all_tmp
    all_dRi = multiples
    all_dRi[np.isclose(all_dRi, 0)] = 0
    all_QM = S_inv + (S_inv @ all_tmp @ S_inv) / np.repeat(multiples, d * d).reshape(k, d, d)

    for i in range(k):
        dRi = all_dRi[i]  # np.linalg.det(Ri) if denominators[i] > 10 ** (-3) else 0
        c = S[:, i]
        QM = all_QM[i] if dRi else np.eye(d) - np.linalg.pinv(all_Ri[i], hermitian=True) @ all_Ri[i]

        # use local first if all of them fail then solve IP
        sol_val, vec, t = utilZ.localSearch(QM, ones_bound=ones_bound, curr_vec=c, A=A, b=b)
        ip_inputs.append((QM, dRi, vec))

        current_ratio = multiples[i] * (1 + sol_val) if dRi else sol_val / (c @ QM @ c)
        #print(f"{np.all(c==vec)}")
        if current_ratio > best_ratio:
            replace_index = i
            add_vector = vec
            best_ratio = current_ratio
            # max_val = prev_value + np.log(current_ratio)
        # if best_ratio >= 1 + 10 ** (-2):
        #     break

    #print(f"Heursitic: {best_ratio=}")
    if best_ratio >= 1 + 10 ** (-3):
        return best_ratio, replace_index, add_vector, False

    for i in range(k):
        QM, dRi, vec = ip_inputs[i]

        sol_val, vec, t = utilZ.quadIP(QM, start=vec, onesBound=ones_bound, A=A, b=b)

        c = S[:, i]
        current_ratio = multiples[i] * (1 + sol_val) if dRi else sol_val / (c @ QM @ c)

        if current_ratio > best_ratio:
            replace_index = i
            add_vector = vec
            best_ratio = current_ratio

        if best_ratio >= 1 + 10 ** (-3):
            break
    #print(f"IP: {best_ratio=}")
    #print(" ")
    return best_ratio, replace_index, add_vector, True


def reduce_dimension(S, k : int):
    # local_move_fun = partial(local_move, ones_bound=ones_bound)
    start_timer = time.perf_counter()
    d, nmax = S.shape
    # new_arr_s, ldet_arr_s = np.zeros((len(arr_s), d, k)), np.zeros(len(arr_s))

    SS = S @ S.T
    SS_inv = np.linalg.inv(SS)
    ldet_S = np.linalg.slogdet(SS)[1]

    for l in range(nmax - k):
        # for all columns c of S, get c @ SS_inv @ c
        all_cZc = (S.T.dot(SS_inv) * S.T).sum(axis=1)
        all_newldet = np.log(1 - all_cZc) + ldet_S

        best_remove_idx = np.argmax(all_newldet)
        best_cZc = all_cZc[best_remove_idx]
        SS_inv += (1 / (1 - best_cZc)) * SS_inv @ np.outer(S[:, best_remove_idx], S[:, best_remove_idx]) @ SS_inv

        S = np.delete(S, best_remove_idx, axis=1)
        ldet_S = all_newldet[best_remove_idx]

    return S, ldet_S, time.perf_counter() - start_timer


def local_alg(k, d, ones_bound=None, seed=None, starting_S=None, A=None, b=None):
    info = {"d": d, "Total Time": time.perf_counter()}
    print(f"starting {d}")

    S = np.ones((d, 10 * k))

    if A is not None:
        R = np.ones((d, 100 * d))
        R[1:, :] = np.random.randint(low=0, high=2, size=(d - 1, 100 * d))
        feas_cols = np.where(np.all(A @ R <= b.reshape(-1, 1), axis=0))[0]
        S = R[:, feas_cols]
        if np.linalg.matrix_rank(S @ S.T) < d:
            print("Could not find solution with non-zero value")
            return {}
        #print(f"got sol with value {np.linalg.slogdet(S @ S.T)[1] }")
    elif ones_bound is not None:
        utilZ.random_p(S, ones_bound, seed=seed)
        info["Ones Bound"] = ones_bound
    else:
        utilZ.random_b(S, seed=seed)



    init_time = 0
    if starting_S is not None:
        S = starting_S
        ldet_init = np.linalg.slogdet(S @ S.T)[1]
    else:
        S, ldet_init, init_time = reduce_dimension(S, k)

    if np.isclose(ldet_init, 0):
        logging.error(f"Could not find a non-zero starting solution random solutions.")
        return {}

    local_move_fun = partial(local_move, ones_bound=ones_bound,A=A, b=b)
    S, current_val, iterations, ip_iter = general_local_alg(S, ldet_init, local_move_fun)
    # print(ldet_init)

    # could not find a non-zero starting solution
    info["Gurobi"] = ip_iter
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Iterations"] = iterations
    info["Initialization Time"] = init_time
    info["init_objval"] = ldet_init
    info["Final Value"] = current_val
    return info


if __name__ == "__main__":
    d = 28
    k = 2 * d


    gen = np.random.default_rng(seed=0)
    num_constraints = 4
    A = np.zeros((num_constraints, d))
    A[:, 1:] = gen.integers(low=0, high=6, size=(num_constraints, d - 1))
    # A = np.random.randint(low = 0, high = 6, size = (4, d))
    b = (d - 1) * 2.5 * np.ones(num_constraints) / 2

    #print(A)

    # 128.79943
    logging.basicConfig(format='[%(levelname)-2s %(filename)s:%(lineno)d] %(message)s',
                        # datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)

    info = local_alg(k, d, A=A, b=b, seed=0)
    print(info)

    # plt.plot( )

    # print(local_alg_ob(k, d, q))

    # print(local_alg_pairs(28 * 2, 28, P))
    # print(local_alg(52, 26, ones_bound=26 // 3))
