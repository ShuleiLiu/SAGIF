import sys
sys.path.append('..')
import numpy as np
import itertools
import math

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


def latin(N, D, lower_bound, upper_bound):
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # 对样本数据进行拉伸
    lower_bounds = np.array(lower_bound)
    upper_bounds = np.array(upper_bound)
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result

def nbi(n, m):
    """Generates a two layers of reference points with H1 and H2 divisions respectively"""
    H1 = 1
    while binomial_coefficient(H1+m,m-1) <= n:
        H1 = H1 + 1
    W1 = nchoosek(list(range(1, H1 + m)), m - 1) - np.tile(np.arange(0, m - 1), (math.comb(H1 + m - 1, m - 1), 1)) - 1
    W1 = (np.hstack((W1, np.zeros([len(W1), 1])+H1))-np.hstack((np.zeros([len(W1), 1]), W1)))/H1
    if H1 < m:
        H2 = 0
        while binomial_coefficient(H1+m-1,m-1)+binomial_coefficient(H2+m,m-1) <= n:
            H2 = H2 + 1
        if (H2 > 0):
            W2 = nchoosek(list(range(1, H2 + m)), m - 1) - np.tile(np.arange(0, m - 1), (math.comb(H2 + m - 1, m - 1),
                                                                    1)) - 1
            W2 = (np.hstack((W2, np.zeros([len(W2), 1])+H2))-np.hstack((np.zeros([len(W2), 1]), W2)))/H2
            W2 = W2/2+1/(2*m)
            W = np.vstack((W1, W2))
        else:
            W = W1
    else:
        W = W1
    W[W<1e-6] = 1e-6
    return W


def nchoosek(v, k):
    return np.array(list(itertools.combinations(v, k)))

def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k)*math.factorial(n-k))

def pf_linear(size, dim, method='latin'):
    if method == 'latin':
        lb = np.zeros(dim)
        ub = np.zeros(dim) + 1.0
        init_samples = latin(size, dim, lb, ub)
    else:
        init_samples = nbi(size, dim)
    scaling_factor = np.sum(init_samples, axis=1)
    final_samples = init_samples / scaling_factor.reshape(-1,1)
    return final_samples

def pf_convex(size, dim, method='latin'):
    if method == 'latin':
        lb = np.zeros(dim)
        ub = np.zeros(dim) + 1.0
        init_samples = latin(size, dim, lb, ub)
    else:
        init_samples = nbi(size, dim)
    scaling_factor = np.sum(init_samples, axis=1)
    final_samples = init_samples / scaling_factor.reshape(-1,1)
    final_samples = final_samples ** 2
    return final_samples

def pf_concave(size, dim, method='latin'):
    if method == 'latin':
        lb = np.zeros(dim)
        ub = np.zeros(dim) + 1.0
        init_samples = latin(size, dim, lb, ub)
    else:
        init_samples = nbi(size, dim)
    scaling_factor = np.sum(init_samples, axis=1)
    final_samples = init_samples / scaling_factor.reshape(-1, 1)
    final_samples = final_samples ** 0.5
    return final_samples

def pf_inverted_linear(size, dim, method='latin'):
    if method == 'latin':
        lb = np.zeros(dim)
        ub = np.zeros(dim) + 1.0
        init_samples = latin(size, dim, lb, ub)
    else:
        init_samples = nbi(size, dim)
    scaling_factor = np.sum(init_samples, axis=1)
    final_samples = init_samples / scaling_factor.reshape(-1, 1)
    final_samples = 1.0 - final_samples
    return final_samples


def pf_inverted_convex(size, dim):
    init_samples = []
    for i in range(size):
        one = []
        s = 1.0
        for i in range(dim-1):
            r = np.random.uniform(0, s)
            while (s - r) / (dim - len(one) - 1) < 0:
                r = np.random.uniform(0, s)
            one.append(1.0 - np.sqrt(r))
            s = s - r
        one.append(1.0 - np.sqrt(s))
        np.random.shuffle(one)
        init_samples.append(one)
    init_samples = np.array(init_samples)
    return init_samples


def pf_inverted_concave(size, dim):
    init_samples = []
    for i in range(size):
        one = []
        s = 1.0
        for i in range(dim-1):
            r = np.random.uniform(0, s)
            while (s - r) / (dim - len(one) - 1) < 0:
                r = np.random.uniform(0, s)
            one.append(1.0 - r**2)
            s = s - r
        one.append(1.0 - s**2)
        np.random.shuffle(one)
        init_samples.append(one)
    init_samples = np.array(init_samples)
    return init_samples


def pf_concave_tmp(size, dim):
    init_samples = []
    for i in range(size):
        one = []
        s = 1.0
        for i in range(dim-1):
            r = np.random.uniform(0, 1.0)
            while (s - r) / (dim - len(one) - 1) < 0:
                r = np.random.uniform(0, 1.0)
            one.append(np.sqrt(r))
            s = s - r
        one.append(s**0.5)
        np.random.shuffle(one)
        init_samples.append(one)
    init_samples = np.array(init_samples)
    return init_samples

# 归一化
# data的数据类型为array
def normalization(data):
    min = np.min(data, axis=0)
    min = min * 0.9
    max = np.max(data, axis=0)
    return (data-min)/((max-min)*1.1)
