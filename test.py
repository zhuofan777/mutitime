import numpy as np
def load_raw_ts():
    path = "D://tmppro//data//raw//BasicMotions//"
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
data_train, data_test, target_train, target_test, nclass = load_raw_ts()
# [0][0] 就是一次测试的一维数据
# print(data_train[0][0])
tmp = data_train[0][0]
print(tmp[0:3])
# 计算样本标准差。
# 没有使用总体标准差
print(np.std(tmp[0:3],ddof=1))
# Train size Dims Length