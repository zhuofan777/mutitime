import numpy as np
import heapq as hq


# 生成序列的局部标准差
def gen_std(series, b):
    l = len(series)
    # 窗口下取整
    w = int(l / b)
    # print(w)
    res = []
    for i in range(l):
        # 窗口左右边界
        left = max(0, i - w)
        right = min(l - 1, i + w)
        # 窗口
        to_do = series[left:right + 1]
        res.append(np.std(to_do, ddof=1))
    return res


# 选择L*α最大的点的下标
def gen_points(series, a):
    l = len(series)
    # 生成l*a个
    res = hq.nlargest(int(l * a), range(l), key=lambda x: series[x])
    return res


def gen_keys(series, points):
    l = len(series)
    ptl = len(points)
    res = []
    for i in range(ptl):
        left = max(points[i] - 1, 0)
        right = min(points[i] + 1, l - 1)
        cur = points[i]
        if (series[left] > series[cur] and series[right] > series[cur]) or (
                series[left] < series[cur] and series[right] < series[cur]):
            res.append(cur)
        elif (left == cur and series[right] > series[cur]) or (left == cur and series[right] < series[cur]) or (
                right == cur and series[left] > series[cur]) or (right == cur and series[left] < series[cur]):
            res.append(cur)
    return res


# w为window长度
# 生成一个序列列表
# [[0, 1], [1, 2, 3], [0, 1, 2], [2, 3]]
def gen_window(series, points, w):
    l = len(series)
    ptl = len(points)
    res = []
    for i in range(ptl):
        cur = points[i]
        # 已cur为中心，左右各w的序列
        tmp = [x  for x in range(max(cur - w, 0),min(cur + w + 1, l - 1))]
        if len(tmp) > 1:
            res.append(tmp)
    return res


# a = np.array([1, 2, 3, 4, 5])
# t = gen_std(a, 2)
# t = gen_std(t, 2)
# print(t)
# s = gen_points(t, 1)
# print(s)
# m = gen_keys(t, s)
# print(m)
# t = gen_window(a, m, 1)
# print(t)
