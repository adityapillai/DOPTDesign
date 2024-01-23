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
        new_val, replace_index, new_vec = local_move_fun(S)
        iterations += 1
        if new_val - current_val < 10 ** (-5):
            break
        current_val = new_val
        S[:, replace_index] = new_vec

    return current_val, iterations


def local_move_pairs(S, Pairs):
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

    def local_pairs_fun(S):
        return local_move_pairs(S, Pairs)

    current_val, iterations = general_local_alg(S, current_val, local_pairs_fun)

    info["Final Value"] = np.log(current_val)
    info["Iterations"] = iterations
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    print(info)
    return info


def local_move(S):
    d, k = S.shape
    # starting point
    SS = S @ S.T
    dS = np.linalg.det(SS)
    Sinv = np.linalg.inv(SS)

    # print(S)
    replace_index, add_vector, max_val = 0, 0, 0
    for i in range(k):
        c = S[:, i].reshape((d, 1))
        tmp = c @ c.T

        Ri = SS - tmp
        dRi = np.linalg.det(Ri)

        if dRi < 10 ** (-15) or d == k:
            dRi = 0

        c = c.flatten()
        QM = Sinv + (Sinv @ tmp @ Sinv) / (1 - c @ Sinv @ c) if dRi > 0 else -1 * np.linalg.pinv(Ri,
                                                                                                 hermitian=True) @ Ri

        sol = utilZ.quadIP(QM)
        sol_val = sol[0]

        norm2 = np.array(sol[1]) @ np.array(sol[1])
        current_val = dRi * (1 + sol_val) if dRi > 0 else (dS / (c @ c + S[:, i] @ QM @ S[:, i])) * (norm2 + sol_val)
        if current_val > max_val:
            replace_index = i
            add_vector = sol[1]
            max_val = current_val

    return max_val, replace_index, add_vector


def local_alg(k, d):
    info = {"F/s": (d - 1, k), "Total Time": time.perf_counter()}

    trials = 50
    arr_s = np.ones((trials, d, k))

    utilZ.random_b(arr_s)
    current_val, max_idx = best_starting_sol(arr_s)
    S = arr_s[max_idx]

    current_val, iterations = general_local_alg(S, current_val, local_move)

    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Iterations"] = iterations
    return info


def local_move_ob(S, q):
    scale = False
    d, k = S.shape
    SS = S @ S.T
    dS = np.linalg.det(SS)
    # Sinv = np.linalg.inv(SS)

    SD = np.linalg.pinv(SS, hermitian=True)

    replace_index, add_vector, max_val = 0, 0, 0
    for i in range(k):
        c = S[:, i].reshape((d, 1))
        tmp = c @ c.T
        Ri = SS - tmp
        c = c.flatten()
        # if scale:
        #    Ri = Ri / d
        # print(Ri)
        # print(remove_Matrix(S, d, k, i))
        inv = True
        dRi = np.linalg.det(Ri)

        # if scale:
        #    dRi

        # do new few lines in a cleaner way
        try:
            QM = np.linalg.inv(Ri)
        except np.linalg.LinAlgError as err:
            dRi = 0
            inv = False
            QM = -1 * np.linalg.pinv(Ri, hermitian=True) @ Ri

        sol = utilZ.quadIP(QM, onesBound=q)

        new_sol = sol[0] / d if scale else sol[0]
        new_sol = max(0, new_sol) if inv else new_sol

        # if new_sol < 0 and inv:
        #    print(new_sol)

        d2 = (SD @ c).T @ (SD @ c)

        current_val = dRi * (1 + new_sol) if inv else (dS * d2) * (q + new_sol)

        if current_val > max_val + 10 ** (-5):
            replace_index = i
            add_vector = sol[1]
            max_val = current_val

    return max_val, replace_index, add_vector


def local_alg_ob(k, d, q):
    info = {"F/s": (d - 1, k), "Ones Bound": q, "Total Time": time.perf_counter(), "Iterations": 0}

    # get 50 random solutions and start with the best
    trials = 50
    arr_s = np.ones((trials, d, k))

    for S in arr_s:
        utilZ.random_p(S, q)

    current_val, max_idx = best_starting_sol(arr_s)
    S = arr_s[max_idx]

    def local_move_ob_fun(S):
        return local_move_ob(S, q)

    currentVal, iterations = general_local_alg(S, current_val, local_move_ob_fun)

    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Final Value"] = np.log(currentVal)
    return info


if __name__ == "__main__":
    d = 15
    k = 2 * d
    q = 5
    P = [[1, 5]]

    # print(local_alg_ob(k, d, q))
    local_alg_pairs(k, d, P)
