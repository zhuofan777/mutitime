import argparse
import math
from random import randint

import numpy as np
from sklearn.metrics import accuracy_score

import _mycif
import diffStd

# 通过cif和关键点结合
from lagcor import lag_cor


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


parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=5, help='times to repeat')
args = parser.parse_args()

signF = True


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

    total_cor_matrix = np.zeros(shape=(data_train.shape[1], data_train.shape[1]))

    # 每个测试集
    for s in range(data_train.shape[0]):
        # 每个维度
        key_list = []
        for j in range(data_train.shape[1]):
            # 做两次局部标准差提取和一个最大点提取
            #  b = 100
            a = diffStd.gen_std(data_train[s][j], 60)
            a = diffStd.gen_std(a, 60)
            #  a = 0.4
            points = diffStd.gen_points(data_train[s][j], 0.3)
            keys = diffStd.gen_keys(data_train[s][j], points)
            # list.sort(keys)
            key_list.append(keys)
        relation_matrix = np.zeros(shape=(len(key_list), len(key_list)))
        for j in range(len(key_list)):
            # 一条关键点序列
            ls = key_list[j]
            # 对于每条序列
            for k in range(data_train.shape[1]):
                if j == k:
                    continue
                cntpoint = 0
                # 对于每个关键点
                for m in ls:
                    maxCor = 0
                    # 滞后相关性的max长度范围为window/2，window长度取为3
                    for n in range(-30, 31):
                        # print(data_train[i][j])
                        maxCor = max(maxCor, abs(lag_cor(data_train[s][j], data_train[s][k], m, 60, n)))
                    y = 0.7
                    if maxCor > y:
                        flag = False
                        for c in range(max(0, m - 60), min(data_train.shape[2], m + 60)):
                            if c in key_list[k]:
                                flag = True
                                break
                        if flag:
                            cntpoint += 1
                # print(cntpoint, ls)
                if len(ls) == 0:
                    continue
                relation_matrix[j][k] = cntpoint / len(ls)
        total_cor_matrix += relation_matrix
    total_cor_matrix /= data_train.shape[0]
    dim_pool = []
    for s in range(total_cor_matrix.shape[0]):
        for j in range(total_cor_matrix.shape[1]):
            if s == j:
                dim_pool.append(s)
                dim_pool.append(j)
            if total_cor_matrix[s][j] >= 0.6:
                dim_pool.append(s)
                dim_pool.append(j)
    # print(dim_pool)
    # print(ans)
    dim_pool.sort()
    # 用于输出相关矩阵
    ans = isUseful(dim_pool)
    # print(ans)
    if not ans:
        global signF
        signF = False
        return
    if ans and times == 2:
        s = 'D://tmppro//colrelation//1.txt'
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
    if ans:
        clf = _mycif.NewCanonicalIntervalForest(dim_pool=dim_pool)
        clf.fit(data_train, target_train)
        _mycif.NewCanonicalIntervalForest(...)
        y_pred = clf.predict(data_test)
        sc = accuracy_score(target_test, y_pred)
        print("trained,accuracy is:")
        print(sc)
        print("saving score")
        # path to save
        s = 'D://tmppro//colrelation//' + str(times) + '.csv'
        f = open(s, 'a')
        f.write(args.dataset + ',' + str(sc) + ',' + '\n')
        f.close()
        print("score saved")


for i in range(0, args.times):
    if signF:
        run(i)
