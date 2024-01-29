import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import numpy as np

def getPyomoModelBARON(F,d,pairs):
    model = pyo.ConcreteModel()
    model.nCols = pyo.Param(default=d)
    model.nVars = pyo.Param(default=F)
    model.M = pyo.RangeSet(model.nCols)
    model.N = pyo.RangeSet(model.nVars)
    model.x1 = pyo.Var(model.N, within=pyo.NonNegativeIntegers, bounds = (0,2)) # L = 3
    model.x = pyo.Var(model.M, within=pyo.NonNegativeReals)
    model.constr = pyo.ConstraintList()
    opt = SolverFactory('baron')
    model.constr.add(model.x[1] == 1)
    for i in range(1,2*F + 1):
        if i <= F:
            model.constr.add(model.x[i+1] == model.x1[i])
        else:
            model.constr.add(model.x[i+1] == model.x1[i-F]**2)
    count_pair = 2*F + 1
    for (i,j) in pairs:
        count_pair += 1
        model.constr.add(model.x[count_pair] == model.x1[i+1]*model.x1[j+1])
    model.obj = pyo.Objective(rule = 1,sense = pyo.maximize)
    return model,opt

def o_rule(G,d,model):
    expr = 0
    for i in range(d):
        for j in range(d): # i think we can improve this because of the symmetry
            expr += model.x[i+1]*G[i,j]*model.x[j+1]
    return expr

def runBARON(G,F,d,model,opt):
    time_IP = time.perf_counter()
    model.del_component(model.obj)
    model.obj = pyo.Objective(rule = o_rule(G,d,model),sense = pyo.maximize)
    opt.solve(model)
    newcolS = np.zeros((F+1,1))
    for i in model.x1:
        newcolS[i] = pyo.value(model.x1[i]) 
    # newcolQ = np.zeros(d)
    # for i in model.x:
    #     newcolQ[i] = pyo.value(model.x[i])
    return pyo.value(model.obj),newcolS,time.perf_counter() - time_IP


# model.obj = pyo.Objective(rule = o_rule(W,m,model),sense = pyo.maximize)
# opt.solve(model)
# vector_values = {i: pyo.value(model.x[i]) for i in model.x}
# print("The optimized max values of x are:", vector_values)
# # model.o = pyo.Objective(rule = o_rule(W,m,model),sense = pyo.minimize)
# # opt.solve(model)
# W = np.random.rand(m,m)
# W = W@W.T


# opt.solve(model)
# vector_values = {i: pyo.value(model.x[i]) for i in model.x}
# print("The optimized min values of x are:", vector_values)

