import math
import random
import numpy as np


# nput is double-dimension matrix
# 1-st dim is dim
# 2-st dim is length
def create_window(s_length):
    window_list = []
    tmp = 0
    while tmp < s_length:
        tmp_w = math.sqrt(s_length)
        if tmp + tmp_w >= s_length:
            tmp_w = s_length - tmp
        window_list.append(tmp_w)
        tmp += tmp_w
        # print(tmp_w)
    if window_list[-1] == 1:
        i = -2
        while window_list[i] < 10:
            window_list[i] += 1
            break
        else:
            i += -1
        window_list.pop()
    # print(window_list)
    return window_list


def split_mt(window_list, matrix):
    ans = []
    for i in range(matrix.shape[0]):
        cnt = 0
        c_window_list = window_list
        tmp_l = []
        for j in c_window_list:
            tmp_wl = []
            while j > 0:
                tmp_wl.append(matrix[i][cnt])
                j -= 1
                cnt += 1
            tmp_l.append(np.array(tmp_wl))
        ans.append(np.array(tmp_l))
    ans = np.array(ans)
    return ans
#
# m = [[1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4,
#       4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 7],
#      [1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4,
#       4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 7],
#      [1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4,
#       4, 5, 6, 6, 1, 2, 3, 4, 5, 5, 6, 1, 4, 4, 4, 5, 6, 6, 7]]
# m = np.array(m)
# rd_wd = create_window(53)
# wd_mt = split_mt(rd_wd, m)
#
# print(create_window(np.array(m)))
