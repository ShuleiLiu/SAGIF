import sys
sys.path.append('..')
import numpy as np
import hvwfg as hv_cal
import algorithm.Operators as ope
from surrogate.RBFN import RBFN
import csv
import time

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

def random_sampling(indx, size):
    idx = np.random.choice(indx, size, replace=False)
    return idx

def normalization(data):
    min = np.min(data, axis=0)
    min = min * 0.9
    max = np.max(data, axis=0)
    return (data-min)/((max-min)*1.1)

## L1-norm distance-based rough filter
def flitter_L1(indx, objs, current_fit, k):
    dis = np.zeros(len(indx))
    # 依次计算未选中个体到选中个体之间的距离
    i = 0
    for ix in indx:
        tmp = np.linalg.norm(current_fit - objs[ix, :], ord=1, axis=1)
        dis[i] = np.sum(tmp)
        i = i + 1
    # 排序，从大到小
    soted_indx = np.argsort(-np.array(dis))
    filter_id = indx[soted_indx[0:k]]
    return filter_id


# the exact method to calculate HVC
def get_exact_hvc(fits, cur_fit, ref):
    hvs = np.zeros(len(fits))
    for si in range(len(fits)):
        hvs[si] = hv_cal.wfg(np.vstack((cur_fit, fits[si])), ref)
    return hvs


def sagif(objs, ref, path, pro, tc, sampling_factor=0.035):
    start = time.time()
    running_time = 0.0
    # filter factor
    filter_factor = 0.7
    n = objs.shape[0]
    m = objs.shape[1]
    # sampling_factor = 0.035
    k = 100
    current_fit = []
    sel_indx = []
    indx = np.arange(0, n)
    # 标记精确求解还是使用代理模型
    flag = False
    while len(sel_indx) < k or running_time >= tc:
        hvs = []
        if flag == True:
            hvs = model.predict(np.round(objs[left_indx, :], 7))
        else:
            for i in indx:
                hv = np.prod(ref-objs[i])
                hvs.append(np.abs(hv))
        # 模型预测出的最优个体
        max_indx = np.argmax(hvs)
        if len(current_fit) == 0:
            max_id = indx[max_indx]
            sel_indx.append(max_id)
            indx = np.delete(indx, max_indx)
            current_fit = np.array([objs[max_id, :]])
            current_hv = np.max(hvs)
            flag = True
        else:
            max_id = left_indx[max_indx]
            sample_max_hv = np.max(exact_hvs)
            # 重新采样，真实计算模型预测的最大HVC
            re_eva_hvc = get_exact_hvc(np.array([objs[max_id, :]]), current_fit, ref)
            # 如果模型找到的HVC没有采样的HVC大
            if re_eva_hvc < sample_max_hv:
                # 将需要选择的个体赋值为采样得到的最大HVC
                sample_max_indx = np.argmax(exact_hvs)
                max_id = sample_indx[sample_max_indx]
                re_eva_hvc = sample_max_hv
            current_fit = np.vstack((current_fit, objs[max_id, :]))
            sel_indx.append(max_id)
            max_indx = np.where(indx == max_id)[0][0]
            indx = np.delete(indx, max_indx)
            current_hv = re_eva_hvc
        end = time.time()
        running_time = end - start
        if len(sel_indx) >= k or running_time >= tc:
            break
        else:
            # 过滤之后的大小
            flitter_size = int(len(indx) * filter_factor)
            # 根据欧式距离过滤个体
            filtter_indx = flitter_L1(indx, objs, current_fit, flitter_size)
            # 采样大小
            sample_size = int(len(indx)*sampling_factor)
            sample_indx = random_sampling(filtter_indx, sample_size)
            left_indx = list(set(filtter_indx) - set(sample_indx))
            sample_obj = np.round(objs[sample_indx, :], 7)
            exact_hvs = get_exact_hvc(sample_obj, current_fit, ref)
            exact_hvcs = exact_hvs - current_hv
            labels = ope.normalization(exact_hvcs)
            model = RBFN(m, sample_size)
            model.fit(sample_obj, np.round(labels, 7))
    objs_gi = objs[sel_indx, :]
    hv = hv_cal.wfg(objs_gi, ref)
    file_name = path + 'sagifr_' + pro +'_' + str(m) +'.csv'
    with open(file_name, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([hv, len(sel_indx), running_time] + sel_indx)
    return running_time
