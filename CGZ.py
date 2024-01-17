import utilZ
import program
import numpy as np
import time
import mosek
import random
import math
import pyipopt
import pyknitro
from collections import defaultdict



def randCol(d):
    vec = d*[1]
    for i in range(d):
        if random.random() > 1/2:
            vec[i] *= 0
    return vec


def randCol_p(d, p):
    vec = d*[0]
    indices = random.sample(range(d), p)

    for i in indices:
        vec[i] = 1

    return vec



def newRandomCols(S, n):

    d = S.shape[0]

    new = np.zeros((0, 0))
    for i in range(n):
        #rand = randCol(d - 1)
        col = [1] + randCol(d - 1)
        while utilZ.colExist(S, np.array(col).reshape(d, 1)):
            col = [1] + randCol(d-1)
        if new.shape[1] > 0:
            new = np.append(new, np.array(col).reshape(d, 1), axis = 1)
        else:
            new = np.array(col).reshape(d, 1)
        #print(S)
    return new


def newRandomCols_l(S, n):

    d = S.shape[0]

    new = np.zeros((0, 0))
    for i in range(n):
        col = np.ones((d, 1))
        col[1:,:] = np.random.randint(low = 0, high = 2, size = (d-1, 1))
        #[1] + randCol(d - 1)
        while utilZ.colExist(S, col):
            col[1:,:] = np.random.randint(low = 0, high = 2, size = (d-1, 1))

        new = np.append(new, np.array(col).reshape(d, 1), axis = 1) if new.shape[1] > 0 else np.array(col).reshape(d, 1)

    return new



# random vectors with at at most p ones, where first coordinate of each vector is 1
def newRandomCols_p(S, n, p):

    d = S.shape[0]

    new = np.zeros((0, 0))
    for i in range(n):
        #rand = randCol(d - 1)
        col = [1] + randCol_p(d - 1, random.randint(0, p- 1))

        attempts = 0
        while utilZ.colExist(S, np.array(col).reshape(d, 1)) and attempts <= 5:
            col = [1] + randCol_p(d - 1, random.randint(0, p- 1))
            attempts += 1
            #print(col)
        if new.shape[1] > 0:
            new = np.append(new, np.array(col).reshape(d, 1), axis = 1)
        else:
            new = np.array(col).reshape(d, 1)
        #print(S)


    return new




# sparsify columns with mosek by solving primal if W is not given
def sparsify_columns(S, k, W = None):
    totalPrimal = 0
    totalSparse = 0
    d = S.shape[0]
    prevSize= S.shape[1]
    remove = []

    if W is None:
        with mosek.Task() as task2:
            prevSize = S.shape[1]

            t = time.perf_counter()
            program.primalSetup(d, k, task2)
            val2, W = program.primaladdCols(S, k, task2)
            totalPrimal += time.perf_counter() - t

    t = time.perf_counter()
    sparseW = utilZ.sparsify(S, W, k)
    totalSparse += time.perf_counter() - t


    if len(sparseW) > 0:
        remove = [i for i in range(S.shape[1]) if sparseW[i] < 10**(-3)]
        S = np.delete(S, remove, 1)

        newSize = S.shape[1]
        #print(f"previous size {prevSize} -> new size {newSize}, removed {len(remove)}")
    else:
        print("failed to sparsify")


    return S, totalPrimal, totalSparse






def colgen_dual(d, k):
    start = time.perf_counter()

    #local = localZ.localAlg(k, d)
    #S = local[1]


    S = np.ones((d, 1))

    optValue = 0
    cols = S.shape[1]
    iter = 0
    #print("starting")
    totalIP = 0
    totalDual = 0
    intialVal = 0

    #print("starting task")
    S = newRandomCols(S, d*(d-1))
    cols = S.shape[1]
    #np.append(S, newRandomCols_p(S, 2*utilZ.c2(d), p), axis = 1)
    with mosek.Task() as task:
        totalDual += program.dualSetUp(d, k, task)
        #print("finished setup")

        G, nu, value, dual_time = program.dualAddCols(S, task)
        totalDual +=  dual_time
        intialVal = value

        while True:

            #ip_time = time.perf_counter()
            obj_IP, col_IP, ip_time  = utilZ.quadIP(G)
            totalIP += IP_time


            newCol = np.array(col_IP).reshape((d,1))

            if utilZ.colExist(S, newCol) or obj_IP <= nu + 10**(-5):
                optValue = value
                break

            S = np.append(S,newCol, axis = 1)

            # d- 1
            newCols = newRandomCols(S, d-1)
            S = np.append(S, newCols, axis = 1)

            cols += d
            #cols += 1

            G, nu, value, dual_time = program.dualAddCols(np.append(newCols, newCol, axis = 1), task)
            #
            totalDual+= dual_time
            #print(value)

            iter += 1

    #print("initial value, final value, iterations, columns")
    #print([intialVal, optValue, iter, cols])
    #print(" ")
    #print("IP Time , MOSEK TIME")
    #print([totalIP, totalDual])
    #print(" ")
    return value


# run column generation where each vector was at most q ones and first coodinate of each vector is 1
def colgen_p(d, k, q, solver = "IPOPT"):

    info = defaultdict(float)
    info["F/s"] = (d-1, k)
    info["Ones Bound"] = q
    info["Times Sparsified"] = 0
    info["Mosek Iterations"] = 0
    #info = {"F/s" : (d-1, k), "Ones Bound" : q }
    solverKey = f"{solver} Time"
    primal_solver = pyipopt.run_ipopt if solver == "IPOPT" else pyknitro.run_knitro

    info["Total Time"] = time.perf_counter()

    # start with a random solution with non-zero value
    S = np.ones((d, k))

    utilZ.random_p(S, d, k, q)

    while np.linalg.det(S @ S.T) < 10**(-5):
        utilZ.random_p(S, d, k, q)


    sparseThrshold = d**2//3
    #

    # add d random columns
    newCols = newRandomCols_p(S, d, q)
    optValue = 0
    iter = 1
    runMosek = False
    mosekIter = 0



    lastIPTime = 0
    currSolver = solver

    cols = S.shape[1]



    # solve IPOPT releaxation with initial set of vectors
    v, weights, lastSolverTime = primal_solver(S.T, k)
    info[solverKey] += lastSolverTime
    value = -v


    # compute solution to dual from primal
    G = np.linalg.inv(S @ np.diag(weights) @ S.T)
    nu = np.max((S.T.dot(G)*S.T).sum(axis=1))

    intialVal = value


    while True:

        obj_IP, col_IP, lastIPTime = utilZ.quadIP(G, onesBound =  q)
        info["Gurobi Time"] +=  lastIPTime
        #totalIP += lastIPTime

        #print(f"itearation {iter}: value is {value}, IP Time {lastIPTime}, {currSolver} time {lastSolverTime}")

        newCol = np.array(col_IP).reshape((d,1))

        if utilZ.colExist(S, newCol) or obj_IP <= nu + 10**(-5):
            optValue = value
            break

        S = np.append(S,newCol, axis = 1)


        numRandCols = 2*(d - 1)**2 if runMosek else d-1
        newCols = newRandomCols_p(S, numRandCols, q)
        S = np.append(S, newCols, axis = 1)

        G = 0

        if not runMosek:
        # compute solution to dual from primal
            v, weights, lastSolverTime = primal_solver(S.T, k)
            info[solverKey] += lastSolverTime
            # switch to mosek if increase was too small from prveious iteration
            runMosek = (-v - value)/value < 0.000001


            value = -v
            G = np.linalg.inv(S @ np.diag(weights) @ S.T)
            nu = np.max((S.T.dot(G)*S.T).sum(axis=1))

        else:
            currSolver = "MOSEK"
            info["Mosek Iterations"] += 1
            weights = None
            with mosek.Task() as task:
                lastSolverTime = program.dualSetUp(d, k, task)
                G, nu, value, dTime = program.dualAddCols(S, task)
                lastSolverTime += dTime
            info["Mosek Time"] +=  lastSolverTime


        # sparsify if 1 number of rows exceeds threshold and we are not running mosek
        if not runMosek and S.shape[1] > sparseThrshold:

            S, totalPrimal, tS = sparsify_columns(S, k, W = weights)

            info["Times Sparsified"] += 1
            info["Sparse Time"] += tS

        iter += 1


    info["Total Time"] = time.perf_counter() - info["Total Time"]
    info["Total Iterations"] = iter
    info["CP Value"] = optValue


    return info

# variant for fixed number of pairs with {0, 1} entries for vectors
def colgen_pairs(d, k, Pairs, solver = "IPOPT"):

    info = defaultdict(float)
    info["F/s"] = (d-1, k)
    info["Pairs"] = Pairs
    info["Times Sparsified"] = 0
    info["Mosek Iterations"] = 0
    #info = {"F/s" : (d-1, k), "Ones Bound" : q }
    solverKey = f"{solver} Time"
    primal_solver = pyipopt.run_ipopt if solver == "IPOPT" else pyknitro.run_knitro

    info["Total Time"] = time.perf_counter()

    primal_solver = pyipopt.run_ipopt if solver == "IPOPT" else pyknitro.run_knitro

    # start with a random solution with non-zero value
    S = np.ones((d, k))

    utilZ.random_b(S)
    pm = utilZ.pairsMat(S, Pairs)


    while np.linalg.det(pm @ pm.T) < 10**(-5):
        utilZ.random_b(S)
        pm = utilZ.pairsMat(S, Pairs)

    #print(f"starting with intergral sol of value {np.linalg.det(pm @ pm.T)}")
    sparseThrshold = d**2//3
    #

    # add d random columns
    newCols = newRandomCols(S, d)
    optValue = 0
    cols = S.shape[1]
    iter = 1
    runMosek = False
    mosekIter = 0


    intialVal = 0
    lastIPTime = 0
    lastSolverTime = 0
    #sp = True

    cols = S.shape[1]

    pm = utilZ.pairsMat(S, Pairs)
    lastSolver = solver
    # solve IPOPT releaxation with initial set of vectors
    v, weights, lastSolverTime = primal_solver(pm.T, k)
    value = -v
    info[solverKey] = lastSolverTime


    # compute solution to dual from primal
    G = np.linalg.inv(pm @ np.diag(weights) @ pm.T)
    nu = np.max((pm.T.dot(G)*pm.T).sum(axis=1))




    intialVal = value

    while True:
        #print(f"itearation {iter}: value is {value}")
        obj_IP, col_IP, lastIPTime = utilZ.quadIP(G, pairs = Pairs)
        info["Gurobi Time"] += lastIPTime

        #print(f"Itearation {iter}: value is {value}, IP Time {lastIPTime}, {lastSolver} time {lastSolverTime}")


        newCol = np.array(col_IP).reshape((d,1))

        if utilZ.colExist(S, newCol) or obj_IP <= nu + 10**(-5):
            optValue = value
            break

        S = np.append(S,newCol, axis = 1)

        numRandCols = 2*(d - 1)**2 if runMosek else d - 1
        newCols = newRandomCols(S, numRandCols)
        S = np.append(S, newCols, axis = 1)



        pm = utilZ.pairsMat(S, Pairs)

        G = 0

        if not runMosek:
        # compute solution to dual from primal
            v, weights, lastSolverTime = primal_solver(pm.T, k)
            runMosek = value > 10**(-3) and (-v - value)/value < 0.000001
            value = -v
            #Q = pm @ np.diag(f(np.maximum(weights, np.zeros(weights.shape[0]))))
            G = np.linalg.inv(pm @ np.diag(weights) @ pm.T)
            nu = np.max((pm.T.dot(G)*pm.T).sum(axis=1))
            info[solverKey] += lastSolverTime
        else:
            lastSolver = "MOSEK"
            mosekIter += 1
            weights = None
            with mosek.Task() as task:
                lastSolverTime = program.dualSetUp(d + len(Pairs), k, task)
                G, nu, value, dTime = program.dualAddCols(pm, task)
                # need primal weights also
            lastSolverTime += dTime
            info["Mosek Time"] +=  lastSolverTime




        # sparsify if number of rows exceeds threshold and we are not running mosek
        if not runMosek and S.shape[1] > sparseThrshold:

            pm, totalPrimal, tS = sparsify_columns(pm, k, W = weights)
            S = pm[0: d , :]

            info["Times Sparsified"] += 1
            info["Sparse Time"] += tS


        iter += 1


    info["Total Time"] = time.perf_counter()  - info["Total Time"]
    info["Total Iterations"] = iter
    info["CP Value"] = optValue


    return info

# same as colgen_pairs except vectors can take entries in {0, 1, 2}
def colgen_levels(d, k, Pairs, solver = "IPOPT"):

    start = time.perf_counter()

    primal_solver = pyipopt.run_ipopt if solver == "IPOPT" else pyknitro.run_knitro

    # start with a random solution with non-zero value
    S = np.ones((d, k))

    utilZ.random_l(S)
    pm = utilZ.pairsMat(S, Pairs)


    while np.linalg.det(pm @ pm.T) < 10**(-5):
        utilZ.random_l(S)
        pm = utilZ.pairsMat(S, Pairs)

    #print(f"starting with intergral sol of value {np.linalg.det(pm @ pm.T)}")
    sparseThrshold = d**2//3
    #

    # add d random columns
    newCols = newRandomCols_l(S, d)
    optValue = 0
    cols = S.shape[1]
    iter = 1
    runMosek = False
    mosekIter = 0


    # timing and other statistics
    totalIP = 0
    totalSparse = 0
    totalDual = 0
    intialVal = 0
    sparseCount = 0
    mosekTime = 0
    #sp = True

    lastIPTime = 0
    lastSolverTime = 0
    lastSolver = solver

    cols = S.shape[1]

    pm = utilZ.pairsMat(S, Pairs)

    # solve IPOPT releaxation with initial set of vectors
    v, weights, t = primal_solver(pm.T, k)
    #pyipopt.run_ipopt(pm.T, k)
    lastSolverTime = t
    # pyipopt.run_ipopt(S.T, k)
    value = -v
    totalDual += t


    # compute feasible solution to dual from primal
    G = np.linalg.inv(pm @ np.diag(weights) @ pm.T)

    nu = np.max((pm.T.dot(G)*pm.T).sum(axis=1))
    #max([pm[:, i] @ G @ pm[:, i] for i in range(pm.shape[1])])


    target = nu


    intialVal = value

    # last iteration that sparsification was run

    while True:

        separator = "Local Search"

        obj_S, col_S, time_S = utilZ.localSearch(G, Pairs)


        # five percent is threshold for when to use local search versus IP
        if (obj_S - target)/target < 0.05:
            separator = "IP"
            obj_S, col_S, time_S = utilZ.quadIP_two(G, Pairs, start = col_S)
            totalIP += time_S


        print(f"itearation {iter}: value is {value}, {separator} Time {time_S}, {lastSolver} time {lastSolverTime}")


        newCol = np.array(col_S).reshape((d,1))

        if separator == "IP" and utilZ.colExist(S, newCol) or obj_S <= nu + 10**(-5):
            optValue = value
            break

        S = np.append(S,newCol, axis = 1)

        numRandCols = 2*(d - 1)**2 if runMosek else d - 1
        newCols = newRandomCols_l(S, numRandCols)
        S = np.append(S, newCols, axis = 1)


        pm = utilZ.pairsMat(S, Pairs)

        G = 0

        if not runMosek:
        # compute solution to dual from primal
            v, weights, lastSolverTime = primal_solver(pm.T, k)
            # switch to mosek if increase was too small from previous iteration
            runMosek = value > 10**(-3) and (-v - value)/value < 0.000001

            value = -v
            G = np.linalg.inv(pm @ np.diag(weights) @ pm.T)
            nu = np.max((pm.T.dot(G)*pm.T).sum(axis=1))

            target = nu
            totalDual += lastSolverTime
        else:
            lastSolver = "MOSEK"
            mosekIter += 1
            weights = None
            with mosek.Task() as task:
                lastSolverTime = program.dualSetUp(d + len(Pairs), k, task)
                G, nu, value, m_time = program.dualAddCols(pm, task)
                lastSolverTime += m_time
                target = nu

            mosekTime += lastSolverTime


        # sparsify if number of rows exceeds threshold and we are not running mosek
        if not runMosek and S.shape[1] > sparseThrshold:

            pm, totalPrimal, tS = sparsify_columns(pm, k, W = weights)
            S = pm[0: d , :]


            sparseCount += 1
            totalSparse += tS
            lastSparse = iter
            mosekTime += totalPrimal

        iter += 1


    totalTime = time.perf_counter() - start

    subarr_res = [d-1,k,intialVal,optValue,S.shape[1],iter,mosekIter,sparseCount,totalSparse,totalDual,mosekTime,totalIP,totalTime]
    #print(f"initial value {intialVal}, final value {optValue}, iterations {iter}, mosek iterations {mosekIter}, columns {S.shape[1]}")
    #print(f"IP Time  {totalIP}, IPOPT TIME {totalDual}, MOSEK TIME {mosekTime}")
    #print(f"sparsified {sparseCount} times , sparsifying time {totalSparse}")
    #print(f"total time is {totalTime}")
    #print(" ")



    return subarr_res




if __name__ == "__main__":

    P = [(1, 3) ,(4, 5), (6, 14), (8, 17)]
    d = 25
    F = d - 1
    k = 2*d

    #colgen_pairs(d, k, P, solver = "KNITRO")

    #colgen_levels(d, k, P, solver = "KNITRO")
    colgen_p(d, k, d//3, solver = "Knitro")
    '''
    for d in range(20,32):
        F = d - 1
        k = 2*d
        q = d//3

        print(f"F: {F}, k: {k}")
        P = [(1, 3) ,(4, 5), (6, 25), (8, 24)]
        #colgen_primal_pairs_IPOPT(d, k, P)
        colgen_primal_levels_IPOPT(d, k, P)
        print(" ")
    '''
