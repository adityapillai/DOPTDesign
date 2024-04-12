import pandas as pd
import numpy as np
import CGZ
import localZ
from functools import partial
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def ones_alg(d_vals, k_vals, q_vals, local, solver):
    func = localZ.local_alg if local else partial(CGZ.colgen_ones_bound, solver=solver)
    dics = [func(d=d, k=k, ones_bound=q) for d, k, q in zip(d_vals, k_vals, q_vals)]
    return pd.DataFrame(dics)


def pairs_alg(d_vals, k_vals, pairs, local, solver):
    fun = localZ.local_alg_pairs if local else partial(CGZ.colgen_pairs, solver=solver)
    dics = [fun(d=d, k=k, pairs=pairs) for d, k in zip(d_vals, k_vals)]
    return pd.DataFrame(dics)


# d_ranges = np.arange(25,41)
# k_ranges = 2 * d_ranges
# p_ranges = d_ranges

d_ranges = np.arange(11,25)
#d_ranges = np.arange(11,32)
k_ranges = 2 * d_ranges
p_ranges = d_ranges // 3

Pairs = [(1, 3), (4, 5), (6, 14), (8, 17)]

#df_ones = ones_alg(d_ranges, k_ranges, p_ranges, False, solver="Knitro")
#df_ones.to_csv("Data/Ones-CG-020424.csv", index=False)
df_ones_local = ones_alg(d_ranges, k_ranges, p_ranges, True, solver="Knitro")
df_ones_local.to_csv("Data/Ones-LS-020424-gd_free.csv", index=False)

# df_pairs = pairs_alg(d_ranges, k_ranges, Pairs, False, solver="Knitro")
# df_pairs.to_csv("Data/Pairs-CG-v5.csv", index=False)
# df_pairs_loc = pairs_alg(d_ranges, k_ranges, Pairs, True, solver="Knitro")
# df_pairs_loc.to_csv("Data/Pairs-LS-v5.csv", index=False)
