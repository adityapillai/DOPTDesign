import utilZ
import numpy as np
import time
import math
import sys

def remove_Matrix(S, i):
    M = np.delete(S, i, axis = 1)
    return M @ M.T


def localMove_pairs(S, Pairs, oldValue):

    d, k = S.shape
    # starting point
    pm = utilZ.pairsMat(S, Pairs)

    dS = np.linalg.det(pm @ pm.T)
    Sinv = np.linalg.inv(pm @ pm.T)

    #print(S)
    replace_index = 0
    addVector = np.zeros(d)
    maxVal = oldValue
    for i in range(k):
        tmp = remove_Matrix(S, i)
        pmi = utilZ.pairsMat(tmp)

        Ri = pmi @ pmi.T
        dRi = np.linalg.det(Ri)

        if dRi < 10**(-15) or d == k:
            dRi = 0

        c = pm[:, i]
        tmp = np.multiply(c, c.reshape(-1, 1))
        QM = Sinv + (Sinv @ tmp @ Sinv)/(1 - c @ Sinv @ c) if dRi > 0 else -1 * np.linalg.pinv(Ri, hermitian = True) @ Ri


        sol = utilZ.quadIP(QM, pairs = Pairs)

        newSol = sol[0]
        # np.array(sol[0]).reshape((d, 1))

        norm2 = np.array(sol[1]) @ np.array(sol[1])
        currentVal = dRi*(1 + newSol) if dRi > 0 else (dS/(c@c + S[:, i] @ QM @ S[:, i]))*(norm2 + newSol)
        #
        #currentVal = dRi*(1 + m.getObjective().getValue())
        if currentVal > maxVal:
            replace_index = i
            addVector = sol[1]
            maxVal = currentVal

    return maxVal > oldValue, replace_index, addVector, maxVal



def localMove(S, oldValue):

    d = S.shape[0]
    k = S.shape[1]
    # starting point
    dS = np.linalg.det(S @ S.T)
    Sinv = np.linalg.inv(S @ S.T)

    #print(S)
    replace_index = 0
    addVector = np.zeros(d)
    maxVal = oldValue
    for i in range(k):

        Ri = remove_Matrix(S, i)
        dRi = np.linalg.det(Ri)

        if dRi < 10**(-15) or d == k:
            dRi = 0

        c = S[:, i]
        tmp = np.multiply(c, c.reshape(-1, 1))
        QM = Sinv + (Sinv @ tmp @ Sinv)/(1 - c @ Sinv @ c) if dRi > 0 else -1 * np.linalg.pinv(Ri, hermitian = True) @ Ri


        sol = utilZ.quadIP(QM)

        newSol = sol[0]
        # np.array(sol[0]).reshape((d, 1))

        norm2 = np.array(sol[1]) @ np.array(sol[1])
        currentVal = dRi*(1 + newSol) if dRi > 0 else (dS/(c@c + S[:, i] @ QM @ S[:, i]))*(norm2 + newSol)
        #
        #currentVal = dRi*(1 + m.getObjective().getValue())
        if currentVal > maxVal:
            replace_index = i
            addVector = sol[1]
            maxVal = currentVal

    if maxVal > oldValue:
        return [replace_index, addVector, maxVal]
    #print("cannot find better sol")
    return [-1]


def localAlg(k, d):
    #S = [np.array(random_pm(d))[:, np.newaxis] for j in range(k)]

    S = np.ones((d, k))

    utilZ.random_b(S, d, k)
    #print(S)
    #utilZ.gramMatrix(S, d, k)
    while abs(np.linalg.det(S @ S.T)) <= 10**(-5):
        utilZ.random_b(S, d, k)




    currentVal = utilZ.solutionValue(S)
    iterations = 0
    start = time.perf_counter()
    while  True:
        p = localMove(S, currentVal)
        iterations = iterations + 1

        #print(p)
        if len(p) == 1:
            break
        for j in range(d):
            S[j][p[0]] = p[1][j]
        new = p[-1]  #solutionValue(S, d, k)

        if abs(new - currentVal) <= 10**(-5):# or iterations > 2:
            break
        currentVal = new

    end = time.perf_counter() - start
    return [iterations ,S, math.log(currentVal), end]



def localMove_p(S, q, oldValue):

    d, k = S.shape


    scale = False

    GM = S @ S.T



    #if scale:
    #    GM = GM / d

    dS = np.linalg.det(GM)


    Sinv = np.linalg.inv(GM)


    SD =  np.linalg.pinv(GM, hermitian = True)

    replace_index = 0
    addVector = np.zeros(d)
    maxVal = oldValue
    for i in range(k):

        Ri = remove_Matrix(S, i)
        #if scale:
        #    Ri = Ri / d
        #print(Ri)
        #print(remove_Matrix(S, d, k, i))
        inv = True
        dRi = np.linalg.det(Ri)

        #if scale:
        #    dRi

        c = S[:, i]
        tmp = np.multiply(c, c.reshape(-1, 1))
        QM = 0

        try:
            QM = np.linalg.inv(Ri)
        except np.linalg.LinAlgError as err:
            dRi = 0
            inv = False
            QM = -1 * np.linalg.pinv(Ri, hermitian = True) @ Ri


        sol = utilZ.quadIP(QM, onesBound = q)
        #print([sol[0], utilZ.quadIP_p([0]])

        newSol = sol[0]/d if scale else sol[0]




        newSol = max(0, newSol) if inv else newSol

        #if newSol < 0 and inv:
        #    print(newSol)

        d2 = (SD @ c).T @ (SD @ c)

        currentVal = dRi*(1 + newSol) if inv else (dS*d2)*(q + newSol)

        if currentVal > maxVal + 10**(-5):
            replace_index = i
            addVector = sol[1]
            maxVal = currentVal

    return maxVal > oldValue, replace_index, addVector, maxVal


def localAlg_p(k, d, q):

    info = {"F/s" : (d-1, k), "Ones Bound" : q, "Total Time" : time.perf_counter(), "Iterations" : 0 }
    #print([math.log(sys.maxsize), d*math.log(d)])
    # get 50 random solutions and start with the best
    trials = 50
    arr_s = np.ones((trials, d, k))

    for S in arr_s:
        utilZ.random_p(S, q)

    prods = arr_s @ np.transpose(arr_s, axes = (0, 2, 1))
    vals = np.linalg.det(prods)
    max_idx = np.argmax(vals)
    currentVal = vals[max_idx]
    S = arr_s[max_idx]

    #if not scale:
    #    currentVal =  utilZ.solutionValue(S)
    #else:
    #    currentVal = np.linalg.det((S @ S.T)/d)


    #print("starting with " + str(math.log(currentVal) + d*math.log(d)) + "and det was" + str(d**d * np.linalg.det(GM/d)))
    while  True:
        success, replace_index, newVec, newVal = localMove_p(S, q, currentVal)
        info["Iterations"] += 1

        if not success:
            break

        S[:, replace_index] = newVec

        currentVal = newVal

    info["Total Time"] = time.perf_counter() - info["Total Time"]

    #final =

    #if scale:
    #    final += d*math.log(d)

    info["Final Value"] = math.log(currentVal)


    return info




#lo = localAlg_p(8, 6, 2)
'''

for d in range(27,32):
    F = d - 1
    k = 2*d
    q = d//3


    print([F, k, q])
    print("itrations, value, time")
    l = localAlg_p(k, d, q)
    print([l[0], l[2], l[3]])
    print(" ")
'''
