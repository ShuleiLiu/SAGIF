import sys
sys.path.append('..')
import numpy as np
import algorithm.Operators as ope
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


# the non-dominated solution size
n = 200
# ms = [3, 5, 7, 10]
# pfs = ['linear', 'inverted_linear', 'convex', 'inverted_convex', 'concave', 'inverted_concave']
# the number of objectives
ms = [3, 5]
# the shape of Pareto Fronts
pfs = ['linear', 'inverted_linear']
# the number of independent runs
ites = 1
path = 'pf_results/'
# Termination conditions, in seconds
tcs = [[0.6, 1.0, 2.5, 30], [0.6, 1.0, 2.5, 30], [0.6, 1.0, 2.5, 30], [0.6, 1.0, 1.0, 2.0], [0.6, 1.0, 2.5, 30], [0.6, 1.0, 1.0, 2.0]]

for i in range(len(pfs)):
    pf = pfs[i]
    tc_pf = tcs[i]
    for j in range(len(ms)):
        m = ms[j]
        tc_pf_m = tc_pf[j]
        if pf == 'linear':
            func = ope.pf_linear
            ref = np.zeros(m) + 1.15
        elif pf == 'inverted_linear':
            func = ope.pf_inverted_linear
            ref = np.zeros(m) + 1.75
        elif pf == 'convex':
            func = ope.pf_convex
            ref = np.zeros(m) + 1.15
        elif pf == 'inverted_convex':
            func = ope.pf_inverted_convex
            ref = np.zeros(m) + 1.75
        elif pf == 'concave':
            func = ope.pf_concave
            ref = np.zeros(m) + 1.15
        elif pf == 'inverted_concave':
            func = ope.pf_inverted_concave
            ref = np.zeros(m) + 1.75
        for i in range(ites):
            objs = func(n, m)
            sagif(objs, ref, path, pf, tc_pf_m)
    print(pf)



