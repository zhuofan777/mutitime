import argparse

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
# import random_wid
import random_wid_fit
# import random_wid_fit_length

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


parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="FingerMovements",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=5, help='times to repeat')
args = parser.parse_args()


def run(times):
    # loading data
    print("loading data...")
    # load_raw_ts(args.data_path, args.dataset)
    data_train, data_test, target_train, target_test, nclass = load_raw_ts(args.data_path, args.dataset)
    print("data loaded")
    # train
    print("training...")
    # trainsize dim length
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
        # 窗口数
        for i in range(wd_mt.shape[1]):
            tp_m = []
            for j in range(wd_mt.shape[0]):
                tp_m.append(wd_mt[j][i])
            s_cor_matrix = abs(np.corrcoef(tp_m))
            cor_matrix += s_cor_matrix
        # print(cor_matrix)
        cor_matrix /= wd_mt.shape[1]
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
    # judge_cor_matrix 维度*维度
    # 该矩阵就是哪一维度相关就乘以1或0，权重参数设置为1/维度

    # print(judge_cor_matrix)
    for i in range(de_avg_cor_matrix.shape[0]):
        for j in range(i,de_avg_cor_matrix.shape[0]):
            if i == j:
                continue
            if judge_cor_matrix[i][j] == 1:
                judge_cor_matrix[j][i] = 1
            if judge_cor_matrix[j][i] == 1:
                judge_cor_matrix[i][j] = 1

    # print(judge_cor_matrix)

    # CAWPE
    pro_cor_matrix = np.zeros(shape=(judge_cor_matrix.shape[0]))
    for i in range(judge_cor_matrix.shape[0]):
        s = np.sum(judge_cor_matrix[i])
        if s != 0:
            pro_cor_matrix[i] = s / judge_cor_matrix.shape[1]
    # pro_cor_matrix : dimension 1
    # print(pro_cor_matrix)
    # 分类器

    data_train = data_train.transpose((1, 0, 2))
    # data_train : dimension train_size series_length
    data_test = data_test.transpose((1, 0, 2))
    pro_matrix = []
    # 180,6
    # classifier = TimeSeriesForestClassifier()
    classifier = CanonicalIntervalForest()

    for i in range(data_train.shape[0]):
        # 先对每个维度进行训练
        tmp_matrix = np.zeros(shape=(data_train.shape[1], 1, data_train.shape[2]))
        for j in range(data_train.shape[1]):
            length = data_train.shape[2]
            tmpSeries = [pd.Series(data_train[i][j], index=[a for a in range(length)], dtype='float32')]
            tmp_matrix[j] = tmpSeries
        classifier.fit(tmp_matrix, target_train)
        # 训练生成classifier
        tmp_matrix2 = np.zeros(shape=(data_test.shape[1], 1, data_test.shape[2]))
        # tmp_matrix2 : train_size 1 series_length

        for j in range(data_test.shape[1]):
            length = data_test.shape[2]
            tmpSeries = [pd.Series(data_test[i][j], index=[a for a in range(length)], dtype='float32')]
            tmp_matrix2[j] = tmpSeries
        # 生成概率
        pro = classifier.predict_proba(tmp_matrix2)
        # 汇总每个维度的概率
        pro_matrix.append(pro)
        # pro : test-size class
        # print(pro.shape)
    pro_matrix = np.array(pro_matrix)
    # pro_matrix : dimension test-size class
    pro_matrix = pro_matrix.transpose((1, 0, 2))
    # print(pro_matrix)
    # print(pro_matrix)
    final_pro_matrix = []
    # pro_matrix : test-size dimension class
    # 对于每次数据
    for i in range(pro_matrix.shape[0]):
        dim = pro_matrix.shape[1]
        tmp_m = np.zeros(shape=nclass)
        # 计算该次数据的每个维度的概率相加
        for b in range(dim):
            tmp_ma = np.zeros(shape=nclass)
            tmp_ma += pro_matrix[i][b]
            # 遍历每个维度
            for j in range(dim):
                # 判断这两个维度是否相关
                # 相关的话
                if judge_cor_matrix[b][j] == 1:
                    tmp_ma += pro_matrix[i][b] * pro_cor_matrix[b]
            tmp_m += tmp_ma
        final_pro_matrix.append(tmp_m)
    # print(final_pro_matrix)
    final_pro_matrix = np.array(final_pro_matrix)
    lb = np.asarray([[np.argmax(prob)] for prob in final_pro_matrix])
    print("model trained")
    print(args.dataset + "score = ")
    sc = accuracy_score(target_test, lb)
    print(sc)
    print("saving score")

    # path to save
    s = 'D://tmppro//tmp//' + str(times) + '.csv'
    f = open(s, 'a')
    f.write(args.dataset + ',' + str(sc) + ',' + '\n')
    f.close()
    print("score saved")


for i in range(0, args.times):
    run(i)
