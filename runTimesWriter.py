import pandas as pd
import numpy as np
import CGZ








primalSolver = "knitro"

d_ranges = np.arange(25, 41)
k_ranges = 2*d_ranges
p_ranges = d_ranges//3

Pairs = [(1, 3) ,(4, 5), (6, 14), (8, 17)]

# experiment 1 for runtimes of first variant
#variant = "Ones_Bound"
#dics = [CGZ.colgen_p(d, k, p, solver = primalSolver) for d,k,p in zip(d_ranges, k_ranges, p_ranges)]
variant = "pairs"
dics = [CGZ.colgen_pairs(d,k, Pairs, solver = primalSolver) for d,k in zip(d_ranges, k_ranges)]


df = pd.DataFrame(dics)
df.set_index("F/s")
col_order = ["F/s", "Pairs", "Total Iterations", "Total Time", "Gurobi Time", f"{primalSolver} Time", "Mosek Time", "Mosek Iterations", "Times Sparsified", "Sprase Time", "CP Value"]
# reorder columns for better readbility in CSV
df = df[col_order]

df.to_csv(f"{variant}-{primalSolver}.csv", index = False)
