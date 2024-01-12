from gurobipy import *
import numpy as np
import random
import sys
import itertools

def c2(p):
    return math.comb(p, 2)


def colExist(S, newcol):

    for i in range(S.shape[1]):
        current = S[:, i].reshape((S.shape[0], 1))
        if np.array_equal(current, newcol):
            return True

    return False


def random_p(S, d, k, p):
    for j in range(k):
        indices = random.sample(range(1,d), random.randint(0, p-1))
        for i in range(1,d):
            S[i, j] = 1 if i in indices else 0

    #print(S)

def random_b(S):
    S[1:S.shape[0]+1, :] = np.random.binomial(n = 1, p = 1/2, size = (S.shape[0]-1, S.shape[1]))


def random_l(S):
    S[1:S.shape[0] + 1, :] = np.random.randint(low = 0, high = 3, size = (S.shape[0]-1, S.shape[1]))

def solutionValue(S):
    return np.linalg.det(S @ S.T)


def pairsMat(S, pairs):
    newMat = np.zeros((len(pairs), S.shape[1]))

    for col in range(newMat.shape[1]):
        for i, p in enumerate(pairs):
            newMat[i, col] = S[p[0], col] * S[p[1], col]

    return np.vstack([S, newMat])

def prodConstraint(m, a, b, c):
    m.addLConstr(c <= a)
    m.addLConstr(c <= b)
    m.addLConstr(c >= a + b - 1)


def quadFormPairs(pairsS, x, pairs):
    d = pairsS.shape[0] - len(pairs)
    pairsX = pairsMat(x.reshape((d, 1)), pairs).reshape((d + len(pairs)))
    return pairsX @ pairsS @ pairsX



def localMove(pairsS, x, pairs, intitialVal):
    d = len(x)

    x_mat = np.tile(x.reshape((d, 1)), (1, d-1))
    mod_I = np.identity(d)[:, 1:]

    allChoices = np.concatenate([(x_mat + mod_I) % 3, (x_mat + 2*mod_I) % 3], axis = 1)
    allChoices_pairs = pairsMat(allChoices, pairs)

    vals = (allChoices_pairs.T.dot(pairsS)*allChoices_pairs.T).sum(axis=1)
    max_idx = np.argmax(vals)
    max_val = vals[max_idx]

    success = max_val > intitialVal

    return success, max_val, (allChoices[:, max_idx]).flatten()


    '''
    index = replacement = -1
    bestVal = intitialVal
    for i, a, in enumerate(x):
        if not i: continue
        for choice in range(3):
            if choice == a:
                continue
            x[i] = choice
            val = quadFormPairs(pairsS, x, pairs)

            if val > bestVal:
                index, replacement = i, choice
            bestVal = max(bestVal, val)
            x[i] = a

    return index >= 0, index, replacement, bestVal'''

def setStart(m, start, X, B, PB, PX, TB, TX, QB, QX):
    m.params.StartNumber = 0
    d = len(start)

    b_vals = (2*d)*[0]

    for i in range(d):
        if start[i] == 1:
            b_vals[2*i] = 1
        elif start[i] == 2:
            b_vals[2*i] = b_vals[2*i + 1] = 1


    for i in X:
        X[i].Start = start[i]


    for i in range(2*d):
        B[i].Start = b_vals[i]

    for i, j in PB:
        PB[i, j].Start = b_vals[i] * b_vals[j]

    for i, j in PX:
        PX[i, j].Start =  start[i] * start[j]

    for i,j,k in TB:
        TB[i, j, k].Start = b_vals[i] * b_vals[j] * b_vals[k]

    for i,j,k in TX:
        TX[i, j, k].Start = start[i] * start[j] * start[k]

    for i, j, k, l in QB:
        QB[i, j, k, l].Start = b_vals[i] * b_vals[j] * b_vals[k] * b_vals[l]

    for i, j, k, l in QX:
        QX[i, j, k, l].Start = start[i] * start[j] * start[k] * start[l]




def localSearch(pairsS, P):
    d = pairsS.shape[0] - len(P)


    trials = 100
    Y = np.random.randint(low = 0, high = 3, size = (d, trials))
    Y[0, :] = 1
    Y_pairs = pairsMat(Y, P)

    # get all y^T pairsS y for all columns y of Y_pairs
    vals = (Y_pairs.T.dot(pairsS)*Y_pairs.T).sum(axis=1)
    max_idx = np.argmax(vals)
    x = Y[:, max_idx].flatten()
    currVal = vals[max_idx]


    while True:
        success, val, newVec = localMove(pairsS, x, P, currVal)

        if not success:
            break

        currVal = val
        x = newVec

    #print(f"LS found solution of value {currVal}")
    return currVal, x


# let S be a set of vectors in (d, k), meaning columns of S are the vectors in the solution

def sparsify(S, W, k):
    d  = S.shape[0]
    s = S.shape[1]

    m = Model()
    m.Params.LogToConsole = 0


    newW = m.addVars(s, lb = 0, ub = k)
    #m.addMVar((s,), lb = 0, ub = k)
    m.addLConstr(newW.sum() == k)

    #L = S @ np.diag(W) @ S.T
    #R = S @ np.diag(newW) @ S.T

    #m.addConstrs(L[i, j] == R[i, j] for i in range(d) for j in range(i, d))

    for i in range(d):
        for j in range(i,d):
            LHS = quicksum([newW[l]*S[i, l]*S[j, l]  for l in range(s)] )
            RHS = sum([W[l]*S[i, l]*S[j, l]  for l in range(s)] )
            m.addLConstr(LHS == RHS)


    m.optimize()

    return [newW[i].x for i in range(s)] if m.status == GRB.OPTIMAL else []




def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

# maximize x^T A x for x binary and A symmetric
# and first coordinate of x is 1
def quadIP(S, onesBound = None, pairs = []):
    d = S.shape[0] - len(pairs)
    m = Model()



    ind = [(i, j) for i in range(d) for j in range(i + 1, d)]
    #C = np.append(S.diagonal(), 2*S[np.triu_indices(d, k = 1)].flatten())
    l = tuplelist(ind)
    X = m.addVars(d, vtype = GRB.BINARY)
    #  obj = S.diagonal()
    # pairs
    P = m.addVars(l, vtype = 'C', lb = 0, ub = 1)
    # obj =  2*S[np.triu_indices(d, k = 1)].flatten(),

    T = []
    Q = []
    if len(pairs):
        T = m.addVars(tuplelist([(i, j, k) for j,k in pairs for i in range(d)]), vtype = 'C', lb = 0, ub  = 1)
        qind = [(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]) for i in range(len(pairs)) for j in range(i + 1, len(pairs))]
        Q = m.addVars(tuplelist(qind), vtype = 'C', lb = 0, ub  = 1)



    m.addConstr(X[0] == 1)
     # + d*len(P) + c2(d) + c2(len(P))

    for i, j in P:
        prodConstraint(m, X[i], X[j], P[i, j])
        #m.addLConstr(P[i,j] <= X[i])
        #m.addLConstr(P[i,j] <= X[j])
        #m.addLConstr(P[i, j] >= X[i] + X[j] - 1)

    for i,j,k in T:
        if i == j or i == k:
            m.addLConstr(T[i, j, k] == P[j, k])
        else:
            prodConstraint(m, X[i], P[j, k], T[i, j, k])
            #m.addLConstr(T[i, j, k] <= X[i])
            #m.addLConstr(T[i, j, k] <= P[j, k])
            #m.addLConstr(T[i, j, k] >= X[i] + P[j, k] - 1)

    for i,j,k,l in Q:
        if i == k:
            m.addLConstr(Q[i, j, k, l] == T[j, k, l])
        elif j == l:
            m.addLConstr(Q[i, j, k, l] == T[i, k, l])
        else:
            prodConstraint(m, P[i,j], P[k, l], Q[i, j, k, l])
            #m.addLConstr(Q[i, j, k, l] <= P[i, j])
            #m.addLConstr(Q[i, j, k, l] <= P[k, l])
            #m.addLConstr(Q[i, j, k, l] >= P[i, j] +  P[k, l] - 1)



    obj = S.diagonal() @ np.array([X[i] for i in X] + [P[i, j] for i,j in pairs]) + quicksum([2 * P[i, j] *S[i, j] for i,j in P])



    for i in range(d):
        for idx, t in enumerate(pairs):
            p1, p2 = t
            obj += T[i, p1, p2] * 2*S[i, idx + d]

    for i in range(len(pairs)):
        p11, p12 = pairs[i]
        for j in range(i + 1, len(pairs)):
            p21, p22 = pairs[j]
            obj += Q[p11, p12, p21, p22] * 2 * S[i + d, j + d]

    m.setObjective(obj, GRB.MAXIMIZE)


    if onesBound is not None:
        m.addLConstr(X.sum() <= onesBound)
    #m.ModelSense = -1
    m.Params.LogToConsole = 0
    m.optimize()

    sol = [int(X[j].x) for j in range(d)]

    return [m.getObjective().getValue() ,sol]

def quadIP_two(S, pairs, start = None):

    d = S.shape[0] - len(pairs)
    #print(f"intially dim is {d}")
    m = Model()


    X = m.addVars(d, vtype = 'C', lb = 0, ub = 2)
    B = m.addVars(2*d, vtype = 'B')

    m.addConstr(X[0] == 1)

    m.addConstrs((X[i] == B[2*i] + B[2*i + 1] for i in range(d)), name = "xEQ")
    m.addConstrs((B[2*i] >= B[2*i + 1] for i in range(d)), name = "sym")


    PB = m.addVars(tuplelist([(i, j) for i in range(2*d) for j in range(i + 1, 2*d)]), vtype = 'C', lb = 0, ub = 1)


    PX = m.addVars(tuplelist([(i, j) for i in range(d) for j in range(i, d)]), vtype = 'C', lb = 0, ub = 4)

    for i, j in PB:
        prodConstraint(m, B[i], B[j], PB[i, j])



    m.addConstrs(PX[i, j] == PB[2*i, 2*j]  + PB[2*i, 2*j + 1] + PB[2*i + 1, 2*j] + PB[2*i + 1, 2*j + 1] for i,j in PX if i != j)
    m.addConstrs(PX[i, i] == B[2*i] + B[2*i + 1] + 2*PB[2*i, 2*i + 1] for i in range(d))


    tripleInd = []
    tripleX = []
    for i in range(d):
        for j, k in pairs:
            tmp = [(a, b, c) for a in [2*i, 2*i + 1] for b in [2*j, 2*j + 1] for c in [2*k, 2*k + 1]]
            tripleInd.extend(tmp)
            tripleX.append((i, j, k))


    TB = m.addVars(tuplelist(tripleInd), vtype = 'C')
    for i,j, k in TB:
        prodConstraint(m, B[i], PB[j, k], TB[i, j, k])

    TX = m.addVars(tuplelist(tripleX), vtype = 'C', lb = 0, ub = 8)
    for i, j, k in TX:
        RHS = [TB[a, b, c] for a in [2*i, 2*i + 1] for b in [2*j, 2*j + 1] for c in [2*k, 2*k + 1]]
        m.addLConstr(TX[i, j, k] == quicksum(RHS))


    quadInd = []
    quadX = []
    for i in range(len(pairs)):
        t1, t2 = pairs[i]
        for j in range(i, len(pairs)):
            s1, s2 = pairs[j]
            tmp = [(a, b, c, e)  for a in [2*t1, 2*t1 + 1] for b in [2*t2, 2*t2 + 1] for c in [2*s1, 2*s1 + 1] for e in [2*s2, 2*s2 + 1]]
            quadInd.extend(tmp)
            quadX.append((t1, t2, s1, s2))

    QB = m.addVars(tuplelist(quadInd), vtype = 'C')


    for a, b, c, e in QB:
        prodConstraint(m, PB[a, b], PB[c, e], QB[a, b, c, e])


    QX = m.addVars(tuplelist(quadX), vtype = 'C', lb = 0, ub = 16)
    for t1, t2, s1, s2 in quadX:
        RHS = [QB[a, b, c, e]  for a in [2*t1, 2*t1 + 1] for b in [2*t2, 2*t2 + 1] for c in [2*s1, 2*s1 + 1] for e in [2*s2, 2*s2 + 1]]
        m.addLConstr(QX[t1, t2, s1, s2] == quicksum(RHS))

    #start = localSearch(S, pairs)
    if start is not None:
        setStart(m, start, X, B, PB, PX, TB, TX, QB, QX)

    obj = S.diagonal() @ np.array([PX[i, i] for i in range(d)] + [QX[i, j, i, j] for i,j in pairs]) + quicksum([2*PX[i, j]*S[i, j] for i,j in PX if i != j])

    for i in range(d):
        for idx, t in enumerate(pairs):
            p1, p2 = t
            obj += TX[i, p1, p2] * 2*S[i, idx + d]

    for i in range(len(pairs)):
        p11, p12 = pairs[i]
        for j in range(i + 1, len(pairs)):
            p21, p22 = pairs[j]
            obj += QX[p11, p12, p21, p22] * 2 * S[i + d, j + d]

    m.setObjective(obj, GRB.MAXIMIZE)

    #m.Params.LogToConsole = 0
    #m.Params.Heuristics = 0.2
    #m.write("a.lp")
    m.optimize()
    sol = [X[j].x for j in range(d)]

    return [m.getObjective().getValue() ,sol]
