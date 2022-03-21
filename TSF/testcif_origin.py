import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sktime.datasets import load_unit_test

import random_wid_fit
from _mycif import NewCanonicalIntervalForest

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="StandWalkJump",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=1, help='times to repeat')
args = parser.parse_args()

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

    print(data_train.shape)
    total_cor_matrix = []
    for train_size in range(data_train.shape[0]):
        # print(data_train[train_size].shape)
        #     计算两两维度之间的皮尔逊系数
        dim = data_train[train_size].shape[0]
        cor_matrix = abs(np.corrcoef(data_train[train_size]))
        cor_matrix = np.where(cor_matrix >= 0.8, cor_matrix, 0)
        total_cor_matrix.append(cor_matrix)
    total_cor_matrix = np.array(total_cor_matrix)

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
    print(judge_cor_matrix)
    dim_pool = []
    for i in range(judge_cor_matrix.shape[0]):
        for j in range(judge_cor_matrix.shape[1]):
            if judge_cor_matrix[i][j] == 1 :
                dim_pool.append(i)
                dim_pool.append(j)
    if len(dim_pool)==0:
        for i in range(judge_cor_matrix.shape[0]):
            dim_pool.append(i)

    dim_pool.sort()
    # 用于输出相关矩阵
    ans = isUseful(dim_pool)
    # print(ans)
    # if ans:
    #     s = 'D://tmppro//cif_tmp//66.txt'
    #     f = open(s, 'a')
    #     f.write(args.dataset + ',' + str(dim_pool) + ',' + '\n')
    #     f.close()

    # dim_pool = np.array(dim_pool)
    # print(dim_pool)
    # print("correlation computed")
    # # # judge_cor_matrix 维度*维度
    # # # 该矩阵就是哪一维度相关就乘以1或0，权重参数设置为1/维度
    # #
    # #
    # # train
    # print("training...")
    # clf = NewCanonicalIntervalForest(dim_pool=dim_pool)
    # clf.fit(data_train, target_train)
    # NewCanonicalIntervalForest(...)
    # y_pred = clf.predict(data_test)
    # sc = accuracy_score(target_test, y_pred)
    # print("trained,accuracy is:")
    # print(sc)
    # print("saving score")
    #
    # # path to save
    # s = 'D://tmppro//cif_tmp//' + str(times) + '.csv'
    # # s = 'D://tmppro//cif_tmp//' + '0' + '.csv'
    # f = open(s, 'a')
    # f.write(args.dataset + ',' + str(sc) + ',' + '\n')
    # f.close()
    # print("score saved")


for i in range(0, args.times):
    run(i)
