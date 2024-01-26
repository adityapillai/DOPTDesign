import utilZ
import numpy as np
import time


def remove_Matrix(S, i):
    M = np.delete(S, i, axis=1)
    return M @ M.T


def best_starting_sol(arr_s):
    prods = arr_s @ np.transpose(arr_s, axes=(0, 2, 1))
    values = np.linalg.det(prods)
    max_idx = np.argmax(values)
    return values[max_idx], max_idx


def general_local_alg(S, current_val, local_move_fun):
    iterations = 0
    while True:
        new_val, replace_index, new_vec = local_move_fun(S, current_val)
        print(f"Value at {np.log(new_val)}")
        iterations += 1
        if new_val - current_val < 10 ** (-5):
            break
        current_val = new_val
        S[:, replace_index] = new_vec

    return current_val, iterations


def local_move_pairs(S, Pairs, old_val):
    d, k = S.shape
    # starting point
    pm = utilZ.pairsMat(S, Pairs)

    pmpm = pm @ pm.T

    dS = np.linalg.det(pmpm)
    Sinv = np.linalg.inv(pmpm)

    # print(S)
    replace_index, addVector, maxVal = 0, 0, 0
    for i in range(k):
        c = pm[:, i].reshape((d + len(Pairs), 1))
        tmp = c @ c.T

        Ri = pmpm - tmp
        dRi = np.linalg.det(Ri)

        if dRi < 10 ** (-15): dRi = 0

        c = c.flatten()
        QM = Sinv + (Sinv @ tmp @ Sinv) / (1 - c @ Sinv @ c) if dRi > 0 else -1 * np.linalg.pinv(Ri,
                                                                                                 hermitian=True) @ Ri

        sol = utilZ.quadIP(QM, pairs=Pairs)
        newSol = sol[0]

        norm2 = np.array(sol[1]) @ np.array(sol[1])
        currentVal = dRi * (1 + newSol) if dRi > 0 else (dS / (c @ c + pm[:, i] @ QM @ pm[:, i])) * (norm2 + newSol)
        #
        # currentVal = dRi*(1 + m.getObjective().getValue())
        if currentVal > maxVal:
            replace_index = i
            addVector = sol[1]
            maxVal = currentVal

    return maxVal, replace_index, addVector


def local_alg_pairs(k, d, Pairs):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}
    trials = 50
    arr_s = np.ones((trials, d, k))

    utilZ.random_b(arr_s)
    arr_s_pairs = np.array([utilZ.pairsMat(S, Pairs) for S in arr_s])
    current_val, max_idx = best_starting_sol(arr_s_pairs)
    S = arr_s[max_idx]

    def local_pairs_fun(S, old_val):
        return local_move_pairs(S, Pairs, old_val)

    current_val, iterations = general_local_alg(S, current_val, local_pairs_fun)

    info["Final Value"] = np.log(current_val)
    info["Iterations"] = iterations
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    print(info)
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

        c = c.flatten()
        QM = S_inv + (S_inv @ tmp @ S_inv) / denominators[i] if dRi else -np.linalg.pinv(Ri, hermitian=True) @ Ri

        if not dRi or ones_bound is not None:
            ip_inputs.append((QM, 0, None))
            continue

        # use local first if all of them fail then solve IP
        sol_val, vec, t = utilZ.localSearch(QM)
        ip_inputs.append((QM, dRi, vec))

        norm2 = np.array(vec) @ np.array(vec)
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
        norm2 = np.array(vec) @ np.array(vec)
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
    # if current_val < 10 ** (-3):
    #    return None

    def local_move_curr(S, old_value):
        return local_move(S, old_value, ones_bound)

    print(f"Random found {np.log(current_val)}")
    current_val, iterations = general_local_alg(S, current_val, local_move_curr)

    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Iterations"] = iterations
    return info


if __name__ == "__main__":
    d = 15
    k = 2 * d
    q = 5
    P = [[1, 5]]

    # print(local_alg_ob(k, d, q))
    # local_alg_pairs(k, d, P)
    local_alg(52, 26, ones_bound=26 // 3)
