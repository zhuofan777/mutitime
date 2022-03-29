import argparse
import math
import random
import time
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import sktime.utils.plotting as plot
import numpy as np
from sklearn.metrics import accuracy_score

import _mycif
import diffStd

# 通过cif和关键点结合
from sax_tool import computeSax, computeSim

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="ArticularyWordRecognition",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=5, help='times to repeat')
args = parser.parse_args()


# 用来统计
def isUseful(a):
    dic = {}
    for i in a:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
    res = ""
    for key in dic:
        res = key
        break
    res = dic[res]
    for key in dic:
        if dic[key] != res:
            return True
    return False


def run(times):
    def load_raw_ts(path, dataset):
        path = path + "raw//" + dataset + "//"
        # 训练集
        x_train = np.load(path + 'X_train.npy')
        x_train = np.transpose(x_train, axes=(0, 2, 1))
        x_test = np.load(path + 'X_test.npy')
        x_test = np.transpose(x_test, axes=(0, 2, 1))
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')
        labels = np.concatenate((y_train, y_test), axis=0)
        nclass = int(np.amax(labels)) + 1
        return x_train, x_test, y_train.reshape(-1), y_test.reshape(-1), nclass

    print("loading data...")
    data_train, data_test, target_train, target_test, nclass = load_raw_ts(args.data_path, args.dataset)
    res_data_train = data_train
    print("data loaded")
    sample_nums = data_train.shape[0]
    dim_nums = data_train.shape[1]
    series_length = data_train.shape[2]
    dims_cnt_pool = [0] * dim_nums

    # plot series

    # 显示单条
    # plot.plot_series(pd.Series(data_train[0][0]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][1]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][2]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][3]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][4]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][5]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][6]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][7]), x_label="time", y_label="values")
    # plot.plot_series(pd.Series(data_train[0][8]), x_label="time", y_label="values")
    # plt.show()
    # 显示每条
    # for a in range(0,dim_nums):
    #     for n in range(0,sample_nums):
    #         plot.plot_series(pd.Series(data_train[a][n]), x_label="time", y_label="values")
    #         plt.show()

    # reduction
    print("starting dims reduction")
    random.seed(int(time.time()))
    rd_sample = random.choice(data_train)

    sax = computeSax(rd_sample)
    sax = sax.values
    # sax里面是series，要转成ndarray

    cluster = {}
    sax_list = []
    for v in range(0, sax.shape[0]):
        sax_list.append(list(sax[v][0].values))
    for a in range(0, len(sax_list)):
        for b in range(a, len(sax_list)):
            if a == b:
                continue
            sim = computeSim(sax_list[a], sax_list[b])
            # print(sim)
            if sim:
                if a in cluster:
                    cluster[a].append(b)
                else:
                    cluster[a] = [b]
    for v in cluster:
        cluster[v].append(v)
    # print(cluster)
    dim_all = []
    for a in cluster:
        for n in cluster[a]:
            dim_all.append(n)
    dim_choose = []

    for a in range(0, dim_nums):
        if a not in cluster and a not in dim_all:
            dim_choose.append(a)

    for v in cluster:
        dim_choose.append(random.choice(cluster[v]))
    dim_choose = list(set(dim_choose))
    # print(dim_choose)

    # change dataset
    data_train = data_train.transpose((1, 0, 2))
    data_train_tmp = []
    for i in range(0, dim_nums):
        if i in dim_choose:
            data_train_tmp.append(data_train[i])
    data_train = np.array(data_train_tmp)
    data_train = data_train.transpose((1, 0, 2))

    old_to_new = {}
    dim_choose.sort()
    print(dim_choose)
    for i in range(0,len(dim_choose)):
        old_to_new[i] = dim_choose[i]

    # correlation
    print("computing correlation...")
    total_cor_matrix = []

    # 每个测试集
    wd_mt = []
    for i in range(data_train.shape[0]):
        # 每个维度
        window_pool = []
        for j in range(data_train.shape[1]):
            # 做两次局部标准差提取和一个最大点提取
            #  b = 100
            a = diffStd.gen_std(data_train[i][j], 60)
            a = diffStd.gen_std(a, 60)
            #  a = 0.4
            points = diffStd.gen_points(data_train[i][j], 0.3)
            keys = diffStd.gen_keys(data_train[i][j], points)
            windows = diffStd.gen_window(data_train[i][j], keys, 1)
            window_pool += windows
        window_pool = list(set([tuple(t) for t in window_pool]))
        window_pool = [list(v) for v in window_pool]
        wd_mt.append(window_pool)

    # wd_mt train_size,windows_nums,windows_size
    # # print(wd_mt)
    total_cor_matrix = []
    for i in range(data_train.shape[0]):
        window_pool = wd_mt[i]
        wpl = len(window_pool)
        # print(window_pool)
        all_dim_pool = []
        # 维度
        for d in range(data_train.shape[1]):
            # 窗口数
            data_pool = []
            for j in range(wpl):
                window = window_pool[j]
                data_window = []
                # 窗口大小
                for k in window:
                    data_window.append(data_train[i][d][k])
                    # print(data_train[i][d][k])
                data_pool.append(data_window)
            data_pool = np.array(data_pool)
            all_dim_pool.append(data_pool)
        #  维度 * 窗口 * 窗口大小
        all_dim_pool = np.array(all_dim_pool)
        # print(all_dim_pool.shape)

        # 创建一个维度*维度的空矩阵
        cor_matrix = np.zeros(shape=(data_train.shape[1], data_train.shape[1]))

        # 窗口数
        for i in range(all_dim_pool.shape[1]):
            tp_m = []
            for j in range(all_dim_pool.shape[0]):
                tp_m.append(all_dim_pool[j][i])
            s_cor_matrix = abs(np.corrcoef(tp_m))
            cor_matrix += s_cor_matrix
        cor_matrix /= all_dim_pool.shape[1]
        cor_matrix = np.where(cor_matrix >= 0.8, cor_matrix, 0)
        total_cor_matrix.append(cor_matrix)
    total_cor_matrix = np.array(total_cor_matrix)
    #
    sum_cor_matrix = np.zeros(shape=(total_cor_matrix.shape[1], total_cor_matrix.shape[2]))
    for x in range(total_cor_matrix.shape[0]):
        sum_cor_matrix += total_cor_matrix[x]
    # print(sum_cor_matrix.shape)
    avg_cor_matrix = sum_cor_matrix / (sum_cor_matrix.sum() / (sum_cor_matrix.shape[0] * sum_cor_matrix.shape[1]))
    de_avg_cor_matrix = avg_cor_matrix / avg_cor_matrix[0][0]
    judge_cor_matrix = np.zeros(shape=(de_avg_cor_matrix.shape[0], de_avg_cor_matrix.shape[1]))
    for i in range(de_avg_cor_matrix.shape[0]):
        judge_line = de_avg_cor_matrix[i].sum() / de_avg_cor_matrix[i].shape
        for j in range(de_avg_cor_matrix.shape[1]):
            if de_avg_cor_matrix[i][j] >= judge_line:
                judge_cor_matrix[i][j] = 1
            else:
                judge_cor_matrix[i][j] = 0

    # print(judge_cor_matrix)
    dim_pool = []
    for i in range(judge_cor_matrix.shape[0]):
        for j in range(judge_cor_matrix.shape[1]):
            if judge_cor_matrix[i][j] == 1:
                dim_pool.append(i)
                dim_pool.append(j)
    if len(dim_pool) == 0:
        for i in range(judge_cor_matrix.shape[0]):
            dim_pool.append(i)
    dim_pool.sort()
    n_dim_pool = []
    for i in dim_pool:
        n_dim_pool.append(old_to_new[i])
    dim_pool = n_dim_pool
    print(dim_pool)
    # 用于输出相关矩阵
    # ans = isUseful(dim_pool)
    # print(ans)
    if times == 2:
        s = 'D://tmppro//SAX//test//1.txt'
        f = open(s, 'a')
        f.write(args.dataset + ',' + str(dim_pool) + ',' + '\n')
        f.close()

    dim_pool = np.array(dim_pool)
    # print(dim_pool)
    print("correlation computed")
    # # judge_cor_matrix 维度*维度
    # # 该矩阵就是哪一维度相关就乘以1或0，权重参数设置为1/维度
    #
    #
    # train
    print("training...")
    clf = _mycif.NewCanonicalIntervalForest(dim_pool=dim_pool)
    clf.fit(res_data_train, target_train)
    _mycif.NewCanonicalIntervalForest(...)
    y_pred = clf.predict(data_test)
    sc = accuracy_score(target_test, y_pred)
    print("trained,accuracy is:")
    print(sc)
    print("saving score")

    # path to save

    s = 'D://tmppro//SAX//test//' + str(times) + '.csv'
    # s = 'D://tmppro//cif_tmp//' + '0' + '.csv'
    f = open(s, 'a')
    f.write(args.dataset + ',' + str(sc) + ',' + '\n')
    f.close()
    print("score saved")


for i in range(0, args.times):
    run(i)
