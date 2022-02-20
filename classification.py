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
from random_wid import create_window


#
# def float_format(x):
#     if abs(x) >= 1e10 or 0 < abs(x) < 1e-3:
#         return "%e" % x
#     else:
#         return "%.4f" % x


# X_train, y_train = load_UCR_UEA_dataset(name="ArrowHead", return_X_y=True, split="train")
# X_test, y_test = load_UCR_UEA_dataset(name="ArrowHead", return_X_y=True, split="test")
# classifier = TimeSeriesForestClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# <class 'pandas.core.frame.DataFrame'>
# print(type(X_train))
# mat = np.array(X_train)
# print(mat.shape)
# print(type(mat))
# <class 'numpy.ndarray'>
# print(mat[0].shape)
# <class 'numpy.ndarray'>
# print(type(mat[0]))
# <class 'pandas.core.series.Series'>
# print(mat[0][0].shape)

def load_raw_ts(path, dataset):
    path = path + "raw/" + dataset + "/"
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
parser.add_argument('--data_path', type=str, default="/home/ydssx/pythonProject/data/",
                        help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS",  # NATOPS
                        help='time series dataset. Options: See the datasets list')

args = parser.parse_args()


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
    cor_matrix = np.corrcoef(data_train[train_size])
    total_cor_matrix.append(cor_matrix)
total_cor_matrix = np.array(total_cor_matrix)
sum_cor_matrix = np.zeros(shape=(total_cor_matrix.shape[1], total_cor_matrix.shape[2]))
for x in range(total_cor_matrix.shape[0]):
    sum_cor_matrix += abs(total_cor_matrix[x])
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
# 该矩阵就是哪一维度相关就乘以1或0，权重参数设置为1/维度
# print(judge_cor_matrix)

# 分类器

data_train = data_train.transpose((1, 0, 2))
data_test = data_test.transpose((1, 0, 2))
# print(data_train.shape)
# print(nclass)
pro_matrix = []
# 180,6
classifier = TimeSeriesForestClassifier()
# classifier = CanonicalIntervalForest(n_estimators=200)
#

# clf = ShapeletTransformClassifier(
#     estimator=RotationForest(n_estimators=3),
#     n_shapelet_samples=500,
#     max_shapelets=20,
#     batch_size=100,
# )
# clf.fit(X_train, y_train)
for i in range(data_train.shape[0]):
    tmp_matrix = np.zeros(shape=(data_train.shape[1], 1, data_train.shape[2]))
    for j in range(data_train.shape[1]):
        length = data_train.shape[2]
        tmpSeries = [pd.Series(data_train[i][j], index=[a for a in range(length)], dtype='float32')]
        tmp_matrix[j] = tmpSeries
    # print(tmp_matrix)
    classifier.fit(tmp_matrix, target_train)

    tmp_matrix2 = np.zeros(shape=(data_test.shape[1], 1, data_test.shape[2]))
    for j in range(data_test.shape[1]):
        length = data_test.shape[2]
        tmpSeries = [pd.Series(data_test[i][j], index=[a for a in range(length)], dtype='float32')]
        tmp_matrix2[j] = tmpSeries
    pro = classifier.predict_proba(tmp_matrix2)
    pro_matrix.append(pro)
    # (180, 6)
    # print(pro.shape)
pro_matrix = np.array(pro_matrix)
pro_matrix = pro_matrix.transpose((1, 0, 2))
final_pro_matrix = []
for i in range(pro_matrix.shape[0]):
    dim = pro_matrix.shape[1]
    tmp_m = np.zeros(shape=nclass)
    for b in range(dim):
        tmp_ma = np.zeros(shape=nclass)
        for j in range(dim):
            if judge_cor_matrix[b][j] == 1:
                tmp_ma += pro_matrix[i][j]
        tmp_m += tmp_ma
    final_pro_matrix.append(tmp_m)
final_pro_matrix = np.array(final_pro_matrix)
lb = np.asarray([[np.argmax(prob)] for prob in final_pro_matrix])
print("model trained")
print(args.dataset + "score = ")
sc = accuracy_score(target_test, lb)
print(sc)
print("saving score")

# path to save
s = '/home/ydssx/pythonProject/' + '.csv'
f = open(s, 'a')
f.write(str(sc) + ',' + '\n')
f.close()
print("score saved")

# print(pro)
# pro = classifier.predict_proba(data_test[i])
# pro_matrix[i] = (1/data_train.shape[0])*
# print(pro_matrix.shape)
# print(data_train.shape)

# classifier = TimeSeriesForestClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict_proba(X_test)
# print(y_pred.shape)
# print(accuracy_score(y_test, y_pred))




