import numpy as np
import heapq as hq


# 计算方差窗口的大小
def computeWinSize(length, b):
    if 0 < length <= b:
        return 1
    else:
        return length // b


# 计算标准差数组
def computeStdArr(series, b):
    l = len(series)
    # 窗口下取整
    w = computeWinSize(l, b)
    # print(w)
    res = []
    for i in range(w, l - w):
        win_arr = [0 for i in range(2 * w + 1)]
        cnt = 0
        for j in range(i - w, w + i + 1):
            win_arr[cnt] = series[j]
            cnt += 1
        std = np.std(win_arr, ddof=0)
        res.append(std)
    return res
    # for i in range(l):
    #     # 窗口左右边界
    #     left = max(0, i - w)
    #     right = min(l - 1, i + w)
    #     # 窗口
    #     to_do = series[left:right + 1]
    #     # 计算总体的标准差
    #     res.append(np.std(to_do, ddof=0))


# 选择L*α最大的点的下标,，保留下标
def selectLargePoint(series, a):
    l = len(series)
    # 生成l*a个
    res = hq.nlargest(int(l * a), range(l), key=lambda x: series[x])
    return res


# 判断是否为波峰波谷
def isPeakOrValley(std_arr, index, b):
    win_size = computeWinSize(len(std_arr), b)
    flag = False
    if win_size == 1:
        if ((std_arr[index] > std_arr[index - 1] and std_arr[index] > std_arr[index + 1]) or (
                std_arr[index] < std_arr[index - 1] and std_arr[index] < std_arr[index + 1])):
            flag = True
    elif win_size == 2:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] >= std_arr[
            index + 1] and std_arr[index] > std_arr[index + 2])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <=
                 std_arr[
                     index + 1]) and std_arr[index] < std_arr[index + 2]):
            flag = True
    elif win_size == 3:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3]
             and std_arr[index] >= std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[index - 3]) and std_arr[index] <= std_arr[index + 1] and std_arr[index] < std_arr[
                    index + 2] and std_arr[index] < std_arr[index + 3]):
            flag = True
    elif win_size == 4:
        if ((std_arr[index] >= std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3] and std_arr[index] > std_arr[index - 4]
             and std_arr[index] >= std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3] and std_arr[index] > std_arr[index + 4])
                or
                (std_arr[index] <= std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[
                     index - 3] and std_arr[index] < std_arr[index - 4]
                 and std_arr[index] <= std_arr[index + 1] and std_arr[index] < std_arr[index + 2] and std_arr[index] <
                 std_arr[index + 3] and std_arr[index] < std_arr[index + 4])):
            flag = True
    else:
        if ((std_arr[index] > std_arr[index - 1] and std_arr[index] > std_arr[index - 2] and std_arr[index] > std_arr[
            index - 3] and std_arr[index] > std_arr[index - 4] and std_arr[index] > std_arr[index - 5]
             and std_arr[index] > std_arr[index + 1] and std_arr[index] > std_arr[index + 2] and std_arr[index] >
             std_arr[
                 index + 3] and std_arr[index] > std_arr[index + 4] and std_arr[index] > std_arr[index + 5])
                or
                (std_arr[index] < std_arr[index - 1] and std_arr[index] < std_arr[index - 2] and std_arr[index] <
                 std_arr[
                     index - 3] and std_arr[index] < std_arr[index - 4] and std_arr[index] < std_arr[index - 5]
                 and std_arr[index] < std_arr[index + 1] and std_arr[index] < std_arr[index + 2] and std_arr[index] <
                 std_arr[
                     index + 3] and std_arr[index] < std_arr[index + 4] and std_arr[index] < std_arr[index + 5])):
            flag = True
    return flag


# 找到所有合适的区间
def fitPoint(stdA, indexA, b):
    w = computeWinSize(len(stdA), b)
    l = len(indexA)
    res = []
    for i in range(0, l):
        val = indexA[i]
        if val >= w and val < l - w:
            if isPeakOrValley(stdA, val, b):
                res.append(val + 2 * w + 1)
    return res


def extractFeaturePoint(arr, a, b):
    stdArrOne = computeStdArr(arr, b)
    stdArrTwo = computeStdArr(stdArrOne, b)
    largePoint = selectLargePoint(stdArrTwo, a)
    indexArr = fitPoint(stdArrTwo, largePoint, b)
    return indexArr


# def getAllSerKeyPoint(series,a,b):
#     newSeries = [0 for i in  range(len(series)-1)]
#     for t in range(1,len(series)):
#         newSeries[t-1] = series[t]
#     featurePoint = extractFeaturePoint(newSeries,a,b)
#     valfp = []
#     for j in range(0,len(featurePoint)):
#         valfp.append(featurePoint[j])


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
        tmp = [x for x in range(max(cur - w, 0), min(cur + w + 1, l - 1))]
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
