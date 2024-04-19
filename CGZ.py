import utilZ
import program
import numpy as np
import time
import mosek
#import pyipopt
import pyknitro
from collections import defaultdict
from functools import partial
import logging


def new_random_cols_general(S, n, fun):
    d, start_cols = S.shape
    allCols = np.ones((d, 5 * n))
    fun(allCols)

    for i in range(allCols.shape[1]):
        curr = allCols[:, i].reshape(-1, 1)

        if not utilZ.col_exist(S, curr):
            S = np.append(S, curr, axis=1)

        if S.shape[1] - start_cols == n:
            break
    return S


def new_random_cols(S, n):
    return new_random_cols_general(S, n, utilZ.random_b)


def new_random_cols_levels(S, n):
    return new_random_cols_general(S, n, utilZ.random_l)


# random vectors with at at most p ones, where first coordinate of each vector is 1
def new_random_cols_ones(S, n, p):
    func = partial(utilZ.random_p, p=p)
    return new_random_cols_general(S, n, func)


# todo sparsify columns with mosek by solving primal if W is not given
def sparsify_columns(S, k, W=None):
    totalPrimal = 0
    totalSparse = 0
    d, prevSize = S.shape

    if W is None:
        with mosek.Task() as task2:
            t = time.perf_counter()
            program.primalSetup(d, k, task2)
            val2, W = program.primaladdCols(S, k, task2)
            totalPrimal += time.perf_counter() - t

    t = time.perf_counter()
    sparseW = utilZ.sparsify(S, W, k)
    totalSparse += time.perf_counter() - t

    if len(sparseW):
        remove = [i for i in range(S.shape[1]) if sparseW[i] < 10 ** (-3)]
        S = np.delete(S, remove, 1)

    else:
        logging.warning("Sparsification failed LP was infeasible")

    return S, totalPrimal, totalSparse


def colgen_dual(d, k):
    info = defaultdict(float)
    info["F/s"] = (d - 1, k)

    S = np.ones((d, k))
    utilZ.random_b(S)

    while np.linalg.det(S @ S.T) < 10 ** (-5):
        utilZ.random_b(S)

    iter = 1

    S = new_random_cols(S, d ** 2)

    with mosek.Task() as task:
        t = program.dualSetUp(d, k, task)
        G, nu, value, dual_time = program.dualAddCols(S, task)
        info["Mosek Time"] = t + dual_time

        while True:
            # print(f"Iteration {iter}: Value {value}, columns {S.shape[1]}")
            logging.info(f"Iteration {iter}: value {value}")

            obj_IP, col_IP, ip_time = utilZ.quadIP(G)
            info["Gurobi Time"] += ip_time
            iter += 1

            newCol = col_IP.reshape(-1, 1)

            if utilZ.col_exist(S, newCol) or obj_IP <= nu + 10 ** (-5):
                break

            prev_size = S.shape[1]
            S = np.append(S, newCol, axis=1)
            S = new_random_cols(S, d - 1)

            G, nu, value, dual_time = program.dualAddCols(S[:, prev_size:], task)
            info["Mosek Time"] += dual_time

    info["Final Value"] = value
    info["Iterations"] = iter
    return info


def colgen_pairs(d, k, pairs, levels=False, solver="IPOPT"):
    logging.info(f"Running column generation for {d=},{k=},{len(pairs)} pairs with {solver=}")
    info = {"F/s": (d - 1, k), "Pairs": pairs}
    apply_pairs = partial(utilZ.pairs_mat, pairs=pairs)
    heuristic_ones = partial(utilZ.localSearch, P=pairs, levels=levels)
    ip_func = partial(utilZ.quadIP, pairs=pairs) if not levels else partial(utilZ.quadIP_two, pairs=pairs)
    rand_func = new_random_cols_levels if levels else new_random_cols

    if solver == "IPOPT":
        primal_solver_ones = partial(pyipopt.run_ipopt, s=k)
    else:
        primal_solver_ones = partial(pyknitro.run_knitro, s=k)

    starter_fun = utilZ.random_l if levels else utilZ.random_b

    trials = 50
    arr_s = np.ones((trials, d, k))
    starter_fun(arr_s)

    val, idx = utilZ.best_starting_sol(np.array([apply_pairs(S) for S in arr_s]))
    S = arr_s[idx]

    if val < 10 ** (-5):
        logging.error(f"Could not find non-zero solution in {trials} random solutions")
        return None

    stats = colgen_general(S, apply_pairs, heuristic_ones, ip_func, rand_func, primal_solver_ones)

    stats[f"{solver} Time"] = stats.pop("Primal Solver")
    info.update(stats)
    return info


def colgen_ones_bound(d, k, ones_bound, solver="IPOPT"):
    # define a apply_pairs
    logging.info(f"Running column generation for ones bound variant for {d=},{k=},{ones_bound=} with {solver=}")

    def apply_pairs(S_):
        return S_

    info = {"F/s": (d - 1, k), "Ones Bound": ones_bound}
    heuristic_ones = partial(utilZ.localSearch, ones_bound=ones_bound)
    ip_ones = partial(utilZ.quadIP, onesBound=ones_bound)
    rand_func_ones = partial(new_random_cols_ones, p=ones_bound)

    if solver == "IPOPT":
        primal_solver_ones = partial(pyipopt.run_ipopt, s=k)
    else:
        primal_solver_ones = partial(pyknitro.run_knitro, s=k)

    # get a non-zero starting solution
    trials = 50
    arr_s = np.ones((trials, d, k))
    for S in arr_s:
        utilZ.random_p(S, ones_bound)

    val, idx = utilZ.best_starting_sol(arr_s)
    S = arr_s[idx]

    if val < 10 ** (-5):
        logging.error(f"Could not find non-zero starting solution among {trials} random solutions")
        return None

    logging.debug(f"Starting with a solution with det {val}")
    stats = colgen_general(S, apply_pairs, heuristic_ones, ip_ones, rand_func_ones, primal_solver_ones)

    # print(info)
    stats[f"{solver} Time"] = stats.pop("Primal Solver")
    info.update(stats)
    return info


# todo maybe? make heuristic_fun optional since not necessary
def colgen_general(S, apply_pairs, heuristic_fun, ip_func, new_rand_cols, primal_solver):
    d, k = S.shape
    info = defaultdict(float)
    info["Total Time"] = time.perf_counter()

    sparse_thrshold = d ** 2 // 3
    iter = 1

    S_mat = apply_pairs(S)
    v, weights, lastSolverTime = primal_solver(S_mat.T)
    value = -v
    info["Primal Solver"] = lastSolverTime
    G = np.linalg.inv(S_mat @ np.diag(weights) @ S_mat.T)
    nu = np.max((S_mat.T.dot(G) * S_mat.T).sum(axis=1))
    run_mosek = False
    num_pairs = S_mat.shape[0] - S.shape[0]
    curr_solver = "Primal"

    while True:

        separator = "LS"
        if heuristic_fun is not None:
            obj_sep, col_sep, time_sep = heuristic_fun(G)
        else:
            obj_sep, col_sep, time_sep = -np.inf, None, 0

        if (obj_sep - nu) / nu < 0.05:
            separator = "IP"
            info["IPs solved"] += 1
            obj_sep, col_sep, ip_time = ip_func(G, start=col_sep)
            time_sep += ip_time
            info["Gurobi Time"] += ip_time

        time_sep = f"{time_sep:.1e}" if separator == "LS" else f"{time_sep:.2f}"
        logging.info(
            f"Iteration {iter}: Value {value:.5f}, {separator} time {time_sep}, {curr_solver} time {lastSolverTime:.2f}")

        new_col = col_sep.reshape(-1, 1)

        if separator == "IP" and (utilZ.col_exist(S, new_col) or obj_sep <= nu + 10 ** (-5)):
            break

        S = np.append(S, new_col, axis=1)
        num_rand_cols = 2 * (d - 1) ** 2 if run_mosek else d - 1
        S = new_rand_cols(S, num_rand_cols)
        S_mat = apply_pairs(S)

        if not run_mosek:
            # compute solution to dual from primal
            v, weights, lastSolverTime = primal_solver(S_mat.T)
            info["Primal Solver"] += lastSolverTime
            # switch to mosek if increase was too small from previous iteration
            # todo may need to add condition that value should be somewhat large before switching for levels to work properly
            run_mosek = (-v - value) / value < 0.000001

            value = -v
            G = np.linalg.inv(S_mat @ np.diag(weights) @ S_mat.T)
            nu = np.max((S_mat.T.dot(G) * S_mat.T).sum(axis=1))

        else:
            curr_solver = "Dual"
            info["Mosek Iterations"] += 1
            weights = None
            with mosek.Task() as task:
                lastSolverTime = program.dualSetUp(d + num_pairs, k, task)
                G, nu, value, dTime = program.dualAddCols(S_mat, task)
                lastSolverTime += dTime
            info["Mosek Time"] += lastSolverTime

        # sparsify if 1 number of rows exceeds threshold and we are not running mosek
        # todo may need to add condition to only switch if sep was IP (for levels variant)
        if not run_mosek and S.shape[1] > sparse_thrshold:
            S_mat, totalPrimal, tS = sparsify_columns(S_mat, k, W=weights)
            S = S_mat[0: d, :]

            info["Times Sparsified"] += 1
            info["Sparse Time"] += tS

        iter += 1

    info["Final Value"] = value
    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Iterations"] = iter
    return info


if __name__ == "__main__":
    P = [(1, 3), (4, 5), (6, 14), (8, 17)]
    d = 28
    k = 2 * d
    q = d // 3

    logging.basicConfig(level=logging.DEBUG)
    # print(colgen_pairs(d, k, pairs=P, levels=True))
    print(colgen_ones_bound(d, k, d // 3, solver="Knitro"))