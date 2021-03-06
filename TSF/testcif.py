import argparse
import math
from random import randint

import numpy as np
from sklearn.metrics import accuracy_score
from sktime.datasets import load_unit_test

import random_wid_fit
from _mycif import NewCanonicalIntervalForest

# CIF的

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="BasicMotions",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=5, help='times to repeat')
args = parser.parse_args()


# 用来统计
def isUseful(a):
    dic = {}
    for i in a:
        if i in dic.keys():
            dic[i]+=1
        else:
            dic[i]=1
    res = ""
    for key in dic:
        res = key
        break
    res = dic[res]
    for key in dic:
        if dic[key]!=res:
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
    print("data loaded")
    # correlation
    print("computing correlation...")
    # print(data_train.shape)
    # X_train, y_train = load_unit_test(split="train", return_X_y=True)
    # print(X_train.shape)
    # X_test, y_test = load_unit_test(split="test", return_X_y=True)

    # WARNING!!!!!!!
    # n_instances, n_dimensions, series_length

    # print(data_train.shape)
    total_cor_matrix = []
    for train_size in range(data_train.shape[0]):
        # print(data_train[train_size].shape)
        #     计算两两维度之间的皮尔逊系数
        dim = data_train[train_size].shape[0]
        rd_wd = random_wid_fit.create_window(data_train.shape[1])
        wd_mt = random_wid_fit.split_mt(rd_wd, data_train[train_size])
        # print(wd_mt)
        # print("......................")
        # wd_mt 维度 * (序列被拆分为 shape[1] * 对应长度)

        cor_matrix = np.zeros(shape=(wd_mt.shape[0], wd_mt.shape[0]))
        m = wd_mt.shape[1]
        # rad = randint(int(math.sqrt(m)), m)

        # 窗口数
        # for i in range(wd_mt.shape[1]):
        for i in range(wd_mt.shape[1]):
            tp_m = []
            rad = randint(0, m-1)
            for j in range(wd_mt.shape[0]):
                tp_m.append(wd_mt[j][rad])
            s_cor_matrix = abs(np.corrcoef(tp_m))
            cor_matrix += s_cor_matrix
        # print(cor_matrix)
        cor_matrix /= wd_mt.shape[1]
        # cor_matrix /= rad
        cor_matrix = np.where(cor_matrix >= 0.8, cor_matrix, 0)
        total_cor_matrix.append(cor_matrix)
    print(total_cor_matrix.shape)
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
    # 用于输出相关矩阵
    ans = isUseful(dim_pool)
    # print(ans)
    if ans and times==2:
        s = 'D://tmppro//cif_tmp//44.txt'
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
    clf = NewCanonicalIntervalForest(dim_pool=dim_pool)
    clf.fit(data_train, target_train)
    NewCanonicalIntervalForest(...)
    y_pred = clf.predict(data_test)
    sc = accuracy_score(target_test, y_pred)
    print("trained,accuracy is:")
    print(sc)
    print("saving score")

    # path to save

    s = 'D://tmppro//cif_tmp//' + str(times) + '.csv'
    # s = 'D://tmppro//cif_tmp//' + '0' + '.csv'
    f = open(s, 'a')
    f.write(args.dataset + ',' + str(sc) + ',' + '\n')
    f.close()
    print("score saved")
    #


for i in range(0, args.times):
    run(i)
