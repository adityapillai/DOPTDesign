import mosek
from itertools import chain, combinations
import math
import sys
import numpy as np


def c2(p):
    return math.comb(p, 2)

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()



def indexJ(i, j, d):
    return d*i + j

def indexZ(i, j, d):
    return d**2 + d*i + j

def indexT(i, j, d, s):
    return d**2 + s*d + d*i + j

def indexW(i, d, s):
    return d**2 + 2*s*d + i


def allCols(d):
    R = np.ones((d, 2**d))

    D = [i for i in range(d)]
    col_index = 0
    #print(list(powerset(D)))

    for S in list(powerset(D)):
        #col = np.ones(((d, 1)))
        for j in S:
            R[j][col_index] = -1
        #print(S)
        #print(col)
        col_index += 1

    #print(R)
    return R






def SOC(S, k, task):
    s = S.shape[1]
    d = S.shape[0]
    optvalue = 0
    W = s*[0]

    inf = math.inf
    task.set_Stream(mosek.streamtype.log, streamprinter)
    task.appendvars(d*d + 2*s*d + s + 1)

        # only for the linear constraints
    task.appendcons(d*d + d + 1)


    for i in range(d):
        for j in range(d):
            task.putvarname(indexJ(i, j, d), "J"+str(i)+","+str(j))
            if i != j:
                if j > i:
                    task.putvarbound(indexJ(i, j, d), mosek.boundkey.fx, 0,0)
                else:
                    task.putvarbound(indexJ(i, j, d), mosek.boundkey.fr, -inf,inf)
            else:
                task.putvarbound(indexJ(i, j, d), mosek.boundkey.lo, 0, inf)

    for i in range(s):
        for j in range(d):
            task.putvarname(indexZ(i, j, d), "Z"+str(i)+","+str(j))
            task.putvarname(indexT(i, j, d, s), "T"+str(i)+","+str(j))

    for i in range(s):
        task.putvarname(indexW(i, d, s), "W" + str(i))

    task.putvarname(d*d + 2*s*d + s, "t")

    # first d^2 contraints are VZ = J
    # normal code [sum([S[i, p]*Z[p, j] for p in range(s)]) == J[i, j] for i in range(d) for j in range(d)]
    constraint_num = 0
    for i in range(d):
        for j in range(d):
            asub = [indexZ(p, j, d) for p in range(s)] + [indexJ(i, j, d)]
            aval = [S[i, p] for p in range(s)] + [-1]

            task.putarow(constraint_num, asub, aval)
            task.putconbound(constraint_num, mosek.boundkey.fx, 0, 0)
            constraint_num += 1

    # next contraint index is d^2


    for j in range(d):
        asub = [indexT(i, j, d, s) for i in range(s)] + [indexJ(j, j, d)]
        aval = s*[1] + [-1]
        task.putarow(constraint_num, asub, aval)
        task.putconbound(constraint_num, mosek.boundkey.up, -inf, 0)
        constraint_num += 1


    # sum of weights is 1,
    task.putarow(constraint_num, [indexW(i, d, s) for i in range(s)], s*[1])
    # previous sum was 1
    task.putconbound(constraint_num, mosek.boundkey.fx, k, k)
    constraint_num += 1


    frowIndex = 0
    rquadcone = task.appendrquadraticconedomain(3)
    task.appendafes(3*s*d + d + 1)
    for i in range(s):
        for j in range(d):
            afeidx = [frowIndex, frowIndex + 1, frowIndex + 2]
            varidx = [indexT(i, j, d, s), indexW(i, d, s), indexZ(i, j, d)]
            task.putafefentrylist(afeidx, varidx, [1, 1, math.sqrt(2)])
            task.appendacc(rquadcone, afeidx, None)
            frowIndex += 3

    #        task.putvarbound(indexT(i, j, d, s), boundkey.lo, 0, inf)
    # T >= 0
    task.putvarboundsliceconst(indexT(0, 0, d, s), indexT(s-1, d-1, d, s) + 1, mosek.boundkey.lo, 0, inf)

    # Z variables are free, upper bounds come from constraints
    task.putvarboundsliceconst(indexZ(0, 0, d), indexZ(s-1, d-1, d) + 1, mosek.boundkey.fr, -inf, inf)

    # W between 0 and 1
    task.putvarboundsliceconst(indexW(0, d, s), indexW(s-1, d, s) + 1, mosek.boundkey.ra,0, 1)

    # integer weights
    num = indexW(s-1, d, s) + 1 - indexW(0, d, s)
    task.putvartypelist(range(indexW(0, d, s), indexW(s-1, d, s) + 1), [mosek.variabletype.type_int] * num)

    # t is non-neg
    task.putvarbound(d*d + 2*s*d + s, mosek.boundkey.lo, 0, inf)

    # objective here use last variable
    afeidx = [frowIndex + j for j in range(d)] + [frowIndex + d]
    frowIndex += d + 1
    varidx = [indexJ(i, i, d) for i in range(d)] + [d*d + 2*s*d + s]
    val = (d+1)*[1]

    task.putafefentrylist(afeidx, varidx, val)
    pc = task.appendprimalpowerconedomain(d + 1, d*[1/d])
    task.appendacc(pc, afeidx, None)
    # set coefficent in objective value
    task.putcj(d*d + 2*s*d + s, 1)

    task.putobjsense(mosek.objsense.maximize)
    #task.writedata("first3.ptf")
    task.optimize()

    solsta = task.getsolsta(mosek.soltype.itr)
    xx = task.getxx(mosek.soltype.itr)
    optvalue = d*math.log(xx[-1])
    # add the constant below if weight sum to 1
    #+ d*math.log(k)
    for i in range(s):
        W[i] = xx[indexW(i, d, s)]

    print(W)
    return optvalue, W


# getvarnameindex(somename) -> (asgn,index)
# getarow(i) -> (nzi,subi,vali)
def SOC_addCol(S, k, task):

    d = S.shape[0]
    prev_s = S.shape[1] - 1
    task.appendvars(2*d + 1)
    varIndex = d*d + 2*prev_s*d + prev_s + 1
    W = (prev_s + 1)*[0]
    inf = math.inf

    # Z in (s, d)
    # T in (s, d)
    # W in s

    for j in range(d):
        task.putvarname(varIndex, "Z"+str(prev_s)+","+str(j))
        task.putvarbound(varIndex, mosek.boundkey.fr, -inf, inf)
        task.putvarname(varIndex + 1, "T"+str(prev_s)+","+str(j))
        task.putvarbound(varIndex + 1, mosek.boundkey.lo, 0, inf)
        varIndex += 2

    task.putvarname(varIndex, "W" + str(prev_s))
    task.putvarbound(varIndex, mosek.boundkey.ra, 0, 1)

    constraint_num = 0

    for i in range(d):
        for j in range(d):
            # get constraint here and modfiy
            nzi, subi, vali = task.getarow(constraint_num)
            asgn, zsj = task.getvarnameindex("Z"+str(prev_s) +","+ str(j))
            subi.append(zsj)
            vali.append(S[i, -1])
            task.putarow(constraint_num, subi, vali)
            constraint_num += 1

    for j in range(d):
        asgn, tsj = task.getvarnameindex("T"+str(prev_s) +","+str(j))
        nzi, subi, vali = task.getarow(constraint_num)
        subi.append(tsj)
        vali.append(1)
        task.putarow(constraint_num, subi, vali)
        constraint_num += 1


    asgn, ws = task.getvarnameindex("W"+str(prev_s))
    nzi, subi, vali = task.getarow(constraint_num)
    subi.append(ws)
    vali.append(1)
    task.putarow(constraint_num, subi, vali)
    constraint_num += 1

    frowIndex = 3*prev_s*d + d + 1
    #print(task.getvarnameindex("W"+str(prev_s)))
    asgn, ws = task.getvarnameindex("W"+str(prev_s))

    task.appendafes(3*d)
    rquadcone = task.getaccdomain(0)
    for j in range(d):
        afeidx = [frowIndex, frowIndex + 1, frowIndex + 2]
        asgn, tsj = task.getvarnameindex("T"+str(prev_s) +","+ str(j))
        #print(task.getvarnameindex("T"+str(prev_s) + str(j)))
        asgn, zsj = task.getvarnameindex("Z"+str(prev_s) +","+ str(j))
        #print(task.getvarnameindex("Z"+str(prev_s) + str(j)))

        varidx = [tsj, ws, zsj]
        #print(varidx)

        task.putafefentrylist(afeidx, varidx, [1, 1, math.sqrt(2)])
        task.appendacc(rquadcone, afeidx, None)
        frowIndex += 3

    task.putobjsense(mosek.objsense.maximize)
    #task.writedata("col.ptf")
    task.optimize()


    #solsta = task.getsolsta(mosek.soltype.itr)
    xx = task.getxx(mosek.soltype.itr)
    # assuming last one is geo mean
    asgn, tidx = task.getvarnameindex("t")
    optvalue = d*math.log(xx[tidx]) + d*math.log(k)
    for i in range(prev_s + 1):
        asgn, widx = task.getvarnameindex("W"+str(i))
        W[i] = xx[widx]

    return optvalue, W


def getG(barX, d):
    B = np.zeros((2*d, 2*d))

    p = 0
    for j in range(2*d):
        for i in range(j, 2*d):
            B[i, j] = barX[p]
            B[j, i] = barX[p]
            p += 1

    G = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            G[i, j] = B[i, j]
            G[j, i] = B[i, j]
    return G




def dualAddCols(S, task, keepNo = 0):

    inf = math.inf

    d = S.shape[0]

    row = task.getnumcon()
    #print(f"adding constraints at indices {row + np.arange(S.shape[1])}")
    #print(str(row) + " constraints here")

    task.appendcons(S.shape[1])

    varNum = task.getnumvar()
    if keepNo:
        task.appendvars(keepNo)
        task.putvarboundsliceconst(varNum, varNum + keepNo, mosek.boundkey.lo, 0, inf)

    for i in range(keepNo):
        task.putcj(i + varNum, -1)

    ai, aj = np.triu_indices(d)

    barai = ai.tolist()
    baraj = aj.tolist()



    for k in range(S.shape[1]):
        v = [S[i, k]*S[j, k] for i in range(d) for j in range(i, d)]

        task.putconbound(row, mosek.boundkey.up, -inf, 0)
        task.putaij(row, 0, -1)

        if k < keepNo:
            task.putaij(row, k + varNum, 1)

        s = task.appendsparsesymmat(2*d, baraj, barai,v)
        task.putbaraij(row, 0, [s], [1])
        row += 1



    #task.writedata("update.ptf")
    task.optimize()

    barX = task.getbarxj(mosek.soltype.itr,0)

    xx = task.getxx(mosek.soltype.itr)

    return getG(barX, d), xx[0] ,task.getprimalobj(mosek.soltype.itr)






def dualSetUp(d, num_vec, task):
    inf = math.inf

    BARVARDIM = [2*d]
    task.appendbarvars(BARVARDIM)
    task.appendvars(1 + d)
    task.appendcons(d + 2*c2(d))


    task.putvarname(0, "g")
    task.putvarbound(0, mosek.boundkey.fr, -inf, inf)

    task.putvarboundsliceconst(1, d + 1, mosek.boundkey.fr, -inf, inf)


    # Z is lower triangular and bottom right block is diag
    row = 0
    #for i in range(d):
        #for j in range(i + 1, d):
    for (i, j) in combinations(range(d), 2):

        col = j + d
        sym0 = task.appendsparsesymmat(BARVARDIM[0], [j + d], [i],[1])
        task.putbaraij(row, 0, [sym0], [1.0])


        sym1 = task.appendsparsesymmat(BARVARDIM[0], [col], [i + d], [1])
        task.putbaraij(row + 1, 0, [sym1], [1.0])

        task.putconbound(row, mosek.boundkey.fx, 0, 0)
        task.putconbound(row + 1, mosek.boundkey.fx, 0, 0)
        row += 2



    # exact diagonal entries of bottom right part
    for j in range(d, 2*d):
        sym = task.appendsparsesymmat(BARVARDIM[0], [j, j], [j, j - d], [1, -1/2])
        task.putbaraij(row, 0, [sym], [1])
        task.putconbound(row, mosek.boundkey.fx, 0, 0)
        row += 1



    task.appendafes(2*d + 1)
    expdomain  = task.appendprimalexpconedomain()
    task.putafeg(0, 1)
    frowIndex = 1


    for i in range(d, 2*d):
        s = task.appendsparsesymmat(BARVARDIM[0], [i], [i], [1])
        task.putafebarfentry(frowIndex, 0, [s], [1])

        task.putafefentry(frowIndex + 1, 1 + i - d, 1)
        task.appendacc(expdomain, [frowIndex, 0, frowIndex + 1], None)
        frowIndex += 2

    task.putobjsense(mosek.objsense.minimize)

    task.putcj(0, num_vec)
    task.putcfix(-d)
    task.putclist(list(range(1, d + 1)) , d*[-1])

    #print(f"setup with {task.getnumcon()} constraints")


def primaladdCols(S, num_vec, task, keepNo = 0):
    inf = math.inf
    d = S.shape[0]

    var_count = task.getnumvar()
    task.appendvars(S.shape[1])

    #lo = 1 if keep else 0
    if keepNo > 0:
        task.putvarboundsliceconst(var_count, var_count + keepNo, mosek.boundkey.ra, 1, num_vec)

    # set weights to integer
    #task.putvartypelist(range(var_count, var_count + S.shape[1]), [mosek.variabletype.type_int] * S.shape[1])

    task.putvarboundsliceconst(var_count + keepNo, var_count + S.shape[1], mosek.boundkey.ra, 0, num_vec)

    lambdas = [var_count + i for i in range(S.shape[1])]

    #for i in lambdas:
    #    task.putvarname(i, "L" + str(i - d))

    # should be +2d instead of +d for -1, -1 vectors
    row = 2*c2(d) + d


    for i in range(d):
        for j in range(i):
            nzi, subi, vali = task.getarow(row)
            subi.extend(lambdas)
            vali.extend([-S[i, p]*S[j, p] for p in range(S.shape[1])])
            task.putarow(row, subi, vali)

            row += 1

        # update diagonal entry constraint
        nzi, subi, vali = task.getarow(row)
        subi.extend(lambdas)
        vali.extend((-S[i, :]).tolist())
        task.putarow(row, subi, vali)

        row += 1


    nzi, subi, vali = task.getarow(row)
    subi.extend(lambdas)
    vali.extend(S.shape[1]*[1])
    task.putarow(row, subi, vali)

    #task.writedata("prim.ptf")
    task.optimize()

    #print(task.getprimalobj(mosek.soltype.itr))

    #G = getG(task.getbarxj(mosek.soltype.itr,0), d)
    xx = task.getxx(mosek.soltype.itr)

    #T = np.zeros((d, d))
    L = xx[d:]
    #[xx[d + i] for i in range(S.shape[1])]


    return task.getprimalobj(mosek.soltype.itr), L





def primalSetup(d, num_vec, task):

    inf = math.inf



    BARVARDIM = [2*d]
    task.appendbarvars(BARVARDIM)
    task.appendvars(d)
    task.appendcons(2*d + 3*c2(d) + 1)

    task.putvarboundsliceconst(0, d, mosek.boundkey.fr, -inf, inf)

    row = 0
    #for i in range(d):
        #for j in range(i + 1, d):
    for (i, j) in combinations(range(d), 2):
        col = j + d
        sym0 = task.appendsparsesymmat(BARVARDIM[0], [j + d], [i],[1])
        task.putbaraij(row, 0, [sym0], [1.0])

        sym1 = task.appendsparsesymmat(BARVARDIM[0], [col], [i + d], [1])
        task.putbaraij(row + 1, 0, [sym1], [1.0])

        task.putconbound(row, mosek.boundkey.fx, 0, 0)
        task.putconbound(row + 1, mosek.boundkey.fx, 0, 0)
        row += 2


    for j in range(d, 2*d):
        sym = task.appendsparsesymmat(BARVARDIM[0], [j, j], [j, j - d], [1, -1/2])
        task.putbaraij(row, 0, [sym], [1])
        task.putconbound(row, mosek.boundkey.fx, 0, 0)
        row += 1


    # says that diagonal is equal to k in sum vv^T which is only valid for -1, +1 vectors
     #for i in range(d):
    #    s = task.appendsparsesymmat(BARVARDIM[0], [i], [i],[1])
    #    task.putbaraij(row, 0, [s], [1.0])
    #    task.putconbound(row, mosek.boundkey.fx, num_vec, num_vec)

    #    row += 1

    task.appendafes(2*d + 1)
    expdomain  = task.appendprimalexpconedomain()
    task.putafeg(0, 1)
    frowIndex = 1


    for i in range(d, 2*d):
        s = task.appendsparsesymmat(BARVARDIM[0], [i], [i], [1])
        task.putafebarfentry(frowIndex, 0, [s], [1])

        task.putafefentry(frowIndex + 1, i - d, 1)
        task.appendacc(expdomain, [frowIndex, 0, frowIndex + 1], None)
        frowIndex += 2

    task.putobjsense(mosek.objsense.maximize)
    task.putclist(list(range(0, d)), d*[1])

    # have added 2*c2(d) + 2d constraints before this (for -1, 1 vecotrs), reduce by d for 0, 1 vectors
    count = 0
    #for i in range(d):
    #    for j in range(i):
    for i in range(d):
        for j in range(i):
            s = task.appendsparsesymmat(BARVARDIM[0], [i], [j], [1/2])
            task.putbaraij(row, 0, [s], [1.0])

            task.putconbound(row, mosek.boundkey.fx, 0, 0)
            row += 1
            count += 1

        # RHS of diagonal entry
        s = task.appendsparsesymmat(BARVARDIM[0], [i], [i], [1])
        task.putbaraij(row, 0, [s], [1.0])

        task.putconbound(row, mosek.boundkey.fx, 0, 0)
        row += 1

    #print(count)

    task.putconbound(row, mosek.boundkey.fx, num_vec, num_vec)
    #task.writedata("setup.ptf")
