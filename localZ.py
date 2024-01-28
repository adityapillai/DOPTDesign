import utilZ
import numpy as np
import time
from functools import partial


def best_starting_sol(arr_s):
    prods = arr_s @ np.transpose(arr_s, axes=(0, 2, 1))
    values = np.linalg.det(prods)
    max_idx = np.argmax(values)
    return values[max_idx], max_idx


# todo modify to also retreive heursitic timings and return them
def general_local_alg(S, current_val, local_move_fun):
    iterations = 0
    while True:
        new_val, replace_index, new_vec = local_move_fun(S, current_val)
        iterations += 1
        if new_val - current_val < 10 ** (-5):
            break
        current_val = new_val
        S[:, replace_index] = new_vec

    return current_val, iterations


def local_move_pairs(S, old_val, Pairs):
    d, k = S.shape
    # starting point
    pm = utilZ.pairsMat(S, Pairs)

    pmpm = pm @ pm.T

    dS = np.linalg.det(pmpm)
    Sinv = np.linalg.inv(pmpm)

    # print(S)
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

        ls_val, vec, t = utilZ.localSearch(QM, Pairs)
        ip_inputs.append((QM, dRi, vec))

        norm2 = vec @ vec
        currentVal = dRi * (1 + ls_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (norm2 + ls_val)
        #
        # currentVal = dRi*(1 + m.getObjective().getValue())
        if currentVal > maxVal:
            replace_index = i
            addVector = vec
            maxVal = currentVal

    if maxVal > old_val:
        return maxVal, replace_index, addVector

    for i in range(k):
        QM, dRi, vec = ip_inputs[i]
        c = pm[:, i]

        ip_val, vec, t = utilZ.quadIP(QM, pairs=Pairs, start=vec)
        norm2 = vec @ vec
        currentVal = dRi * (1 + ip_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (norm2 + ip_val)

        if currentVal > maxVal:
            replace_index = i
            addVector = vec
            maxVal = currentVal

    return maxVal, replace_index, addVector

# todo add to this function option for levels, for levels do not solve any IP's and only use heuristic
def local_alg_pairs(k, d, pairs):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}
    trials = 50
    arr_s = np.ones((trials, d, k))

    utilZ.random_b(arr_s)
    arr_s_pairs = np.array([utilZ.pairsMat(S, pairs) for S in arr_s])
    current_val, max_idx = best_starting_sol(arr_s_pairs)
    S = arr_s[max_idx]

    fun = partial(local_move_pairs, Pairs=pairs)

    current_val, iterations = general_local_alg(S, current_val, fun)

    info["Final Value"] = np.log(current_val)
    info["Iterations"] = iterations
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    #print(info)
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

        norm2 = vec @ vec
        current_val = dRi * (1 + sol_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (norm2 + sol_val)

        if current_val > max_val:
            replace_index = i
            add_vector = vec
            max_val = current_val

    if max_val > prev_value:
        return max_val, replace_index, add_vector

    for i in range(k):
        QM, dRi, vec = ip_inputs[i]
        sol_val, vec, t = utilZ.quadIP(QM, start=vec, onesBound=ones_bound)

        c = S[:, i]
        norm2 = vec @ vec
        current_val = dRi * (1 + sol_val) if dRi else (dS / (c @ c + c @ QM @ c)) * (norm2 + sol_val)

        if current_val > max_val:
            replace_index = i
            add_vector = vec
            max_val = current_val

    return max_val, replace_index, add_vector


def local_alg(k, d, ones_bound=None):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}

    trials = 50
    arr_s = np.ones((trials, d, k))

    if ones_bound is None:
        utilZ.random_b(arr_s)
    else:
        for S in arr_s: utilZ.random_p(S, ones_bound)
        info["Ones Bound"] = ones_bound

    current_val, max_idx = best_starting_sol(arr_s)
    S = arr_s[max_idx]

    # could not find a non-zero starting solution
    if current_val < 10 ** (-3):
        return None

    fun = partial(local_move, ones_bound=ones_bound)
    current_val, iterations = general_local_alg(S, current_val, fun)

    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Iterations"] = iterations
    return info


if __name__ == "__main__":
    d = 15
    k = 2 * d
    q = 5
    P = [(1, 3), (4, 5), (6, 14), (8, 17)]

    # print(local_alg_ob(k, d, q))
    local_alg_pairs(80, 40, P)
    # print(local_alg(52, 26, ones_bound=26 // 3))
