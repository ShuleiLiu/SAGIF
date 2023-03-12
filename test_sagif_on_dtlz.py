import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from algorithm.SAGIF import sagif

# ------------------------------- Reference --------------------------------
# S. Liu, H. Wang, W. Yao and W. Peng, "Surrogate-Assisted Environmental Selection for Fast Hypervolume-based Many-Objective Optimization,"
# in IEEE Transactions on Evolutionary Computation, doi: 10.1109/TEVC.2023.3243632.
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2022 HandingWangXD Group. Permission is granted to copy and
# use this code for research, noncommercial purposes, provided this
# copyright notice is retained and the origin of the code is cited. The
# code is provided "as is" and without any warranties, express or implied.
# ------------------------------- Developer --------------------------------
# This code is written by Shulei Liu. Email: shuleiliu@126.com

def random_sel(data, sel_size):
    n = data.shape[0]
    indx = np.arange(0, n)
    sel_indx = np.random.choice(indx, size=sel_size, replace=False)
    sel_data = data[sel_indx, :]
    return sel_data

# normalization
def norm(data):
    min = np.min(data, axis=0)
    min = min
    max = np.max(data, axis=0)
    return (data-min)/((max-min))


# non-dominated solution size
n = 200
# number of objectives
ms = [5]
# test problems
pfs = ['D1', 'D2', 'D7']
nos = ['1', '2', '3']
ites = 1
path = 'dtlz_results/'

for pf in pfs:
    for m in ms:
        for ni in nos:
            print(pf, m, ni)
            tc = np.Inf
            file_name = 'data/' + pf + '_' + str(m) + '_' + ni + '.csv'
            all_data = pd.read_csv(file_name, sep=',', header=None).values
            ref = np.zeros(m) + 1.15
            for i in range(ites):
                objs = random_sel(all_data, n)
                objs = norm(objs)
                sagif(objs, ref, path, pf + '_' + ni, tc)
        print(pf)