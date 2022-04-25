import _sax as sax
import numpy as np

def computeSax(s):
    a = sax.SAX()
    b = a.transform(s)
    return b


def computeSim(m1, m2):
    # print(m1,m2)
    try:
        cnt = int(m1[0] + m1[1] + m1[2] + m1[3])
        a = abs(m1[0] - m2[0])
        b = abs(m1[1] - m2[1])
        c = abs(m1[2] - m2[2])
        d = abs(m1[3] - m2[3])
        s = int(a + b + c + d)
        if s / cnt <= 0.4:
            return True
    except :
        return False

    return False


# def load_raw_ts(path, dataset):
#     path = path + "raw//" + dataset + "//"
#     # 训练集
#     x_train = np.load(path + 'X_train.npy')
#     x_train = np.transpose(x_train, axes=(0, 2, 1))
#     x_test = np.load(path + 'X_test.npy')
#     x_test = np.transpose(x_test, axes=(0, 2, 1))
#     y_train = np.load(path + 'y_train.npy')
#     y_test = np.load(path + 'y_test.npy')
#     labels = np.concatenate((y_train, y_test), axis=0)
#     nclass = int(np.amax(labels)) + 1
#     return x_train, x_test, y_train.reshape(-1), y_test.reshape(-1), nclass
#
#
# print("loading data...")
# data_train, data_test, target_train, target_test, nclass = load_raw_ts("D://tmppro//data//", "NATOPS")
#
# print("data loaded")
#
# print(data_train.shape)
# dim_nums = data_train.shape[1]
#
#
# sample = data_train[0]
# # a.fit(sample)
# b = computeSax(sample)
# # print(b)
# print(b)
