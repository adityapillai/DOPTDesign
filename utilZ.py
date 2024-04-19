from gurobipy import *
import numpy as np
import time
import logging


def col_exist(S, new_col):
    return np.all(new_col == S, axis=0).any()


# todo see if this can be made to work on 3d matrices as well
def random_p(S, p, seed=None):
    d, k = S.shape
    rng = np.random.default_rng(seed=seed)
    S[1:, :] = 0
    numOnes = rng.integers(low=0, high=p, size=k)
    allRows = [rng.choice(d - 1, numOnes[j], replace=False) + 1 for j in range(k)]
    S[np.concatenate(allRows), np.repeat(np.arange(k), numOnes)] = 1
    # for j in range(k):
    #    S[allRows[j], j] = 1


def random_b(S, seed=None):
    rng = np.random.default_rng(seed=seed)
    if S.ndim > 2:
        S[:, 1:S.shape[1] + 1, :] = rng.integers(low=0, high=2, size=(S.shape[0], S.shape[1] - 1, S.shape[2]))
    else:
        S[1:S.shape[0] + 1, :] = rng.integers(low=0, high=2, size=(S.shape[0] - 1, S.shape[1]))


def random_l(S):
    if S.ndim > 2:
        S[:, 1:S.shape[1] + 1, :] = np.random.randint(low=0, high=3, size=(S.shape[0], S.shape[1] - 1, S.shape[2]))
    else:
        S[1:S.shape[0] + 1, :] = np.random.randint(low=0, high=3, size=(S.shape[0] - 1, S.shape[1]))


def best_starting_sol(arr_s):
    prods = arr_s @ np.transpose(arr_s, axes=(0, 2, 1))
    values = np.linalg.det(prods)
    max_idx = np.argmax(values)
    return values[max_idx], max_idx


def pairs_mat(S, pairs):
    newMat = np.zeros((len(pairs), S.shape[1]))

    for col in range(newMat.shape[1]):
        for i, p in enumerate(pairs):
            newMat[i, col] = S[p[0], col] * S[p[1], col]

    return np.vstack([S, newMat])


def prod_constraint(m, a, b, c):
    m.addLConstr(c <= a)
    m.addLConstr(c <= b)
    m.addLConstr(c >= a + b - 1)


# finds matrix for all possible local moves when you swap a 0 and a 1 in the matrix
def local_move_swap(x):
    d, q = len(x), int(x.sum())
    one_idx = np.where(x[1:] == 1)[0]
    mod_I = (np.identity(d)[:, 1:])[:, one_idx]
    mod_I = np.tile(mod_I, d - q)
    zero_idx = np.where(x == 0)[0]
    mod_I[np.repeat(zero_idx, q - 1), np.arange((d - q) * (q - 1))] = 1
    return mod_I


# pairsS, x, pairs, levels
def localMove(pairsS, x, pairs, levels, ones_bound):
    d = len(x)

    mod_I = np.identity(d)[:, 1:]

    if ones_bound is not None and int(x.sum()) == ones_bound:
        indices = np.where(x[1:] == 0)[0]
        mod_I = np.delete(mod_I, indices, axis=1)
        mod_I = np.append(mod_I, local_move_swap(x), axis=1)

    if levels:
        allChoices = np.concatenate([(x.reshape(-1, 1) + mod_I) % 3, (x.reshape(-1, 1) + 2 * mod_I) % 3], axis=1)
    else:
        allChoices = (x.reshape(-1, 1) + mod_I) % 2

    allChoices_pairs = pairs_mat(allChoices, pairs) if len(pairs) else allChoices

    vals = (allChoices_pairs.T.dot(pairsS) * allChoices_pairs.T).sum(axis=1)
    max_idx = np.argmax(vals)
    max_val = vals[max_idx]

    return max_val, (allChoices[:, max_idx]).flatten()


def localSearch(pairsS, P=None, levels=False, ones_bound=None, curr_vec=None):
    # logger =  logging.getLogger(__name__)
    if P is None: P = []
    local_time = time.perf_counter()
    d = pairsS.shape[0] - len(P)

    upper = 3 if levels else 2

    trials = 100 #0 + int(curr_vec is not None)

    Y = np.random.randint(low=0, high=upper, size=(d, trials)) if curr_vec is None else curr_vec.reshape(-1, 1)
    Y[0, :] = 1

    if ones_bound is not None and curr_vec is None:
        random_p(Y, ones_bound)

    Y_pairs = pairs_mat(Y, P) if P else Y

    vals = (Y_pairs.T.dot(pairsS) * Y_pairs.T).sum(axis=1)
    max_idx = np.argmax(vals)
    x = Y[:, max_idx].ravel()
    currVal = vals[max_idx]

    iter = 0
    #print(f"starting with value {currVal} max idx was {max_idx}")
    while True:
        val, newVec = localMove(pairsS, x, P, levels, ones_bound)
        iter += 1

        if val - currVal < 10 ** (-5):
            break

        currVal = val
        x = newVec


    local_time = time.perf_counter() - local_time
    return currVal, x, local_time


# let S be a set of vectors in (d, k), meaning columns of S are the vectors in the solution

def sparsify(S, W, k):
    d, s = S.shape

    m = Model()
    m.Params.LogToConsole = 0

    newW = m.addVars(s, lb=0, ub=k)
    m.addLConstr(newW.sum() == k)

    for i in range(d):
        for j in range(i, d):
            vals = [S[i, l] * S[j, l] for l in range(s)]
            LHS = quicksum([newW[l] * vals[l] for l in range(s)])
            RHS = sum([W[l] * vals[l] for l in range(s)])
            m.addLConstr(LHS == RHS)

    m.optimize()

    return [newW[i].x for i in range(s)] if m.status == GRB.OPTIMAL else []


# maximize x^T A x for x binary and A symmetric
# and first coordinate of x is 1
def quadIP(S, onesBound=None, pairs=None, start=None):
    if pairs is None: pairs = []
    time_IP = time.perf_counter()
    d = S.shape[0] - len(pairs)
    m = Model()

    ind = [(i, j) for i in range(d) for j in range(i + 1, d)]
    l = tuplelist(ind)
    X = m.addVars(d, vtype=GRB.BINARY)
    #  obj = S.diagonal()
    # pairs
    P = m.addVars(l, vtype='C', lb=0, ub=1)

    T = []
    Q = []
    if len(pairs):
        T = m.addVars(tuplelist([(i, j, k) for j, k in pairs for i in range(d)]), vtype='C', lb=0, ub=1)
        qind = [(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]) for i in range(len(pairs)) for j in
                range(i + 1, len(pairs))]
        Q = m.addVars(tuplelist(qind), vtype='C', lb=0, ub=1)

    m.addConstr(X[0] == 1)

    for i, j in P:
        prod_constraint(m, X[i], X[j], P[i, j])

    for i, j, k in T:
        if i == j or i == k:
            m.addLConstr(T[i, j, k] == P[j, k])
        else:
            prod_constraint(m, X[i], P[j, k], T[i, j, k])

    for i, j, k, l in Q:
        if i == k:
            m.addLConstr(Q[i, j, k, l] == T[j, k, l])
        elif j == l:
            m.addLConstr(Q[i, j, k, l] == T[i, k, l])
        else:
            prod_constraint(m, P[i, j], P[k, l], Q[i, j, k, l])

    obj = S.diagonal() @ np.array([X[i] for i in X] + [P[i, j] for i, j in pairs]) + quicksum(
        [2 * P[i, j] * S[i, j] for i, j in P])

    for i in range(d):
        for idx, t in enumerate(pairs):
            p1, p2 = t
            obj += T[i, p1, p2] * 2 * S[i, idx + d]

    for i in range(len(pairs)):
        p11, p12 = pairs[i]
        for j in range(i + 1, len(pairs)):
            p21, p22 = pairs[j]
            obj += Q[p11, p12, p21, p22] * 2 * S[i + d, j + d]

    m.setObjective(obj, GRB.MAXIMIZE)

    if onesBound is not None:
        m.addLConstr(X.sum() <= onesBound)
    if start is not None:
        m.params.StartNumber = 0
        set_start(start, X, P, T, Q)

    m.Params.LogToConsole = 0
    m.optimize()
    time_IP = time.perf_counter() - time_IP

    sol = np.array([int(X[j].x) for j in range(d)])

    return m.getObjective().getValue(), sol, time_IP


def set_start(start, X, P, T, Q):
    for i in X:
        X[i].Start = start[i]

    for i, j in P:
        P[i, j].Start = start[i] * start[j]

    for i, j, k in T:
        T[i, j, k] = start[i] * start[j] * start[k]

    for i, j, k, l in Q:
        Q[i, j, k, l] = start[i] * start[j] * start[k] * start[l]


def set_start_two(m, start, X, B, PB, PX, TB, TX, QB, QX):
    m.params.StartNumber = 0
    d = len(start)

    b_vals = (2 * d) * [0]

    for i in range(d):
        if start[i] == 1:
            b_vals[2 * i] = 1
        elif start[i] == 2:
            b_vals[2 * i] = b_vals[2 * i + 1] = 1

    for i in X:
        X[i].Start = start[i]

    for i in range(2 * d):
        B[i].Start = b_vals[i]

    for i, j in PB:
        PB[i, j].Start = b_vals[i] * b_vals[j]

    for i, j in PX:
        PX[i, j].Start = start[i] * start[j]

    for i, j, k in TB:
        TB[i, j, k].Start = b_vals[i] * b_vals[j] * b_vals[k]

    for i, j, k in TX:
        TX[i, j, k].Start = start[i] * start[j] * start[k]

    for i, j, k, l in QB:
        QB[i, j, k, l].Start = b_vals[i] * b_vals[j] * b_vals[k] * b_vals[l]

    for i, j, k, l in QX:
        QX[i, j, k, l].Start = start[i] * start[j] * start[k] * start[l]


def quadIP_two(S, pairs, start=None):
    time_IP = time.perf_counter()
    d = S.shape[0] - len(pairs)
    m = Model()

    X = m.addVars(d, vtype='C', lb=0, ub=2)
    B = m.addVars(2 * d, vtype='B')

    m.addConstr(X[0] == 1)

    m.addConstrs((X[i] == B[2 * i] + B[2 * i + 1] for i in range(d)), name="xEQ")
    m.addConstrs((B[2 * i] >= B[2 * i + 1] for i in range(d)), name="sym")

    PB = m.addVars(tuplelist([(i, j) for i in range(2 * d) for j in range(i + 1, 2 * d)]), vtype='C', lb=0, ub=1)

    PX = m.addVars(tuplelist([(i, j) for i in range(d) for j in range(i, d)]), vtype='C', lb=0, ub=4)

    for i, j in PB:
        prod_constraint(m, B[i], B[j], PB[i, j])

    m.addConstrs(
        PX[i, j] == PB[2 * i, 2 * j] + PB[2 * i, 2 * j + 1] + PB[2 * i + 1, 2 * j] + PB[2 * i + 1, 2 * j + 1] for i, j
        in PX if i != j)
    m.addConstrs(PX[i, i] == B[2 * i] + B[2 * i + 1] + 2 * PB[2 * i, 2 * i + 1] for i in range(d))

    tripleInd = []
    tripleX = []
    for i in range(d):
        for j, k in pairs:
            tmp = [(a, b, c) for a in [2 * i, 2 * i + 1] for b in [2 * j, 2 * j + 1] for c in [2 * k, 2 * k + 1]]
            tripleInd.extend(tmp)
            tripleX.append((i, j, k))

    TB = m.addVars(tuplelist(tripleInd), vtype='C')
    for i, j, k in TB:
        prod_constraint(m, B[i], PB[j, k], TB[i, j, k])

    TX = m.addVars(tuplelist(tripleX), vtype='C', lb=0, ub=8)
    for i, j, k in TX:
        RHS = [TB[a, b, c] for a in [2 * i, 2 * i + 1] for b in [2 * j, 2 * j + 1] for c in [2 * k, 2 * k + 1]]
        m.addLConstr(TX[i, j, k] == quicksum(RHS))

    quadInd = []
    quadX = []
    for i in range(len(pairs)):
        t1, t2 = pairs[i]
        for j in range(i, len(pairs)):
            s1, s2 = pairs[j]
            tmp = [(a, b, c, e) for a in [2 * t1, 2 * t1 + 1] for b in [2 * t2, 2 * t2 + 1] for c in
                   [2 * s1, 2 * s1 + 1] for e in [2 * s2, 2 * s2 + 1]]
            quadInd.extend(tmp)
            quadX.append((t1, t2, s1, s2))

    QB = m.addVars(tuplelist(quadInd), vtype='C')

    for a, b, c, e in QB:
        prod_constraint(m, PB[a, b], PB[c, e], QB[a, b, c, e])

    QX = m.addVars(tuplelist(quadX), vtype='C', lb=0, ub=16)
    for t1, t2, s1, s2 in quadX:
        RHS = [QB[a, b, c, e] for a in [2 * t1, 2 * t1 + 1] for b in [2 * t2, 2 * t2 + 1] for c in [2 * s1, 2 * s1 + 1]
               for e in [2 * s2, 2 * s2 + 1]]
        m.addLConstr(QX[t1, t2, s1, s2] == quicksum(RHS))

    # start = localSearch(S, pairs)
    if start is not None:
        set_start_two(m, start, X, B, PB, PX, TB, TX, QB, QX)

    obj = S.diagonal() @ np.array([PX[i, i] for i in range(d)] + [QX[i, j, i, j] for i, j in pairs]) + quicksum(
        [2 * PX[i, j] * S[i, j] for i, j in PX if i != j])

    for i in range(d):
        for idx, t in enumerate(pairs):
            p1, p2 = t
            obj += TX[i, p1, p2] * 2 * S[i, idx + d]

    for i in range(len(pairs)):
        p11, p12 = pairs[i]
        for j in range(i + 1, len(pairs)):
            p21, p22 = pairs[j]
            obj += QX[p11, p12, p21, p22] * 2 * S[i + d, j + d]

    m.setObjective(obj, GRB.MAXIMIZE)

    m.Params.LogToConsole = 0
    m.optimize()
    sol = np.array([int(X[j].x) for j in range(d)])

    return m.getObjective().getValue(), sol, time.perf_counter() - time_IP
