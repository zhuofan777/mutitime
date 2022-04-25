import argparse
import collections
import math
import random
import time
from collections import Counter
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import sktime.utils.plotting as plot
import numpy as np
from sklearn.metrics import accuracy_score
import sfa
import _mycif
import diffStd

# 通过cif和关键点结合
from sax_tool import computeSax, computeSim

parser = argparse.ArgumentParser()
# dataset settings
parser.add_argument('--data_path', type=str, default="D://tmppro//data//",
                    help='the path of data.')
# parser.add_argument('--dataset', type=str, default="AtrialFibrilation",  # NATOPS
# parser.add_argument('--dataset', type=str, default="NATOPS",  # NATOPS
parser.add_argument('--dataset', type=str, default="FaceDetection",  # NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--times', type=int, default=5, help='times to repeat')
args = parser.parse_args()


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


def run(times):
    def correlation(data_train, old_to_new):
        # correlation

        total_cor_matrix = []

        # 每个测试集
        wd_mt = []
        for i in range(data_train.shape[0]):
            # 每个维度
            feature_point_list = []
            win = random.choice([1, 2, 3])
            window_pool = []
            for j in range(data_train.shape[1]):
                alp = random.choice([0.3, 0.4])
                blt = random.choice([100, 90, 80, 70, 60, 50])
                feature_point = diffStd.extractFeaturePoint(data_train[i][j], alp, blt)
                window_pool += diffStd.gen_window(data_train[i][j], feature_point, win)
            # print(window_pool)
            window_pool = list(set([tuple(t) for t in window_pool]))
            window_pool = [list(v) for v in window_pool]
            wd_mt.append(window_pool)

        # wd_mt train_size,windows_nums,windows_size
        # # print(wd_mt)
        total_cor_matrix = []
        for i in range(data_train.shape[0]):
            window_pool = wd_mt[i]
            wpl = len(window_pool)
            # print(window_pool)
            all_dim_pool = []
            # 维度
            for d in range(data_train.shape[1]):
                # 窗口数
                data_pool = []
                for j in range(wpl):
                    window = window_pool[j]
                    data_window = []
                    # 窗口大小
                    for k in window:
                        data_window.append(data_train[i][d][k])
                        # print(data_train[i][d][k])
                    data_pool.append(data_window)
                data_pool = np.array(data_pool)
                all_dim_pool.append(data_pool)
            #  维度 * 窗口 * 窗口大小
            all_dim_pool = np.array(all_dim_pool)
            # print(all_dim_pool.shape)

            # 创建一个维度*维度的空矩阵
            cor_matrix = np.zeros(shape=(data_train.shape[1], data_train.shape[1]))

            # 窗口数
            for i in range(all_dim_pool.shape[1]):
                tp_m = []
                for j in range(all_dim_pool.shape[0]):
                    x = all_dim_pool[j][i]
                    # 防止出现None
                    if len(set(x)) == 1:
                        x[0] += 0.000001
                    tp_m.append(np.array(x))
                tp_m = np.array(tp_m)
                s_cor_matrix = abs(np.corrcoef(tp_m))
                cor_matrix += s_cor_matrix
                # print(cor_matrix)
            cor_matrix /= all_dim_pool.shape[1]

            cor_matrix = np.where(cor_matrix >= 0.8, cor_matrix, 0)
            total_cor_matrix.append(cor_matrix)
        total_cor_matrix = np.array(total_cor_matrix)
        # print(total_cor_matrix)
        #
        sum_cor_matrix = np.zeros(shape=(total_cor_matrix.shape[1], total_cor_matrix.shape[2]))
        for x in range(total_cor_matrix.shape[0]):
            sum_cor_matrix += total_cor_matrix[x]
        # print(sum_cor_matrix.shape)
        avg_cor_matrix = sum_cor_matrix / (sum_cor_matrix.sum() / (sum_cor_matrix.shape[0] * sum_cor_matrix.shape[1]))
        # print(avg_cor_matrix)
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
        n_dim_pool = []
        for i in dim_pool:
            n_dim_pool.append(old_to_new[i])
        dim_pool = n_dim_pool
        # print(dim_pool)
        return dim_pool

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
    res_data_train = data_train
    print("data loaded")
    sample_nums = data_train.shape[0]
    dim_nums = data_train.shape[1]
    series_length = data_train.shape[2]
    fake_data_train = []
    fake_dim_choose = []
    dim_choose = []
    # plot series
    # 显示单条
    # plot.plot_series(pd.Series(data_train[0][0]), x_label="time", y_label="values")
    # plt.show()
    # 显示每条
    # for a in range(0,dim_nums):
    #     for n in range(0,sample_nums):
    #         plot.plot_series(pd.Series(data_train[a][n]), x_label="time", y_label="values")
    #         plt.show()
    #  低维 中维 高维
    # 0,1,2
    dim_flag = 0
    if dim_nums < 10:
        # print(data_train.shape)
        #  对每个维度生成一个差分序列
        for s in range(0, sample_nums):
            sample = data_train[s]
            x = []
            for sm in sample:
                length = len(sm)
                # print(sm)
                dif = [0 for i in range(length)]
                for i in range(1, length):
                    dif[i - 1] = sm[i] - sm[i - 1]
                dif[-1] = dif[-2]
                dif = np.array(dif)
                # print(sm,dif)
                x.append(dif)
            # print(x.shape)
            fake_data_train.append(np.array(x))
            # print("---------------")
        fake_data_train = np.array(fake_data_train)
        dim_choose = [i for i in range(0, dim_nums)]
        fake_dim_choose = [i for i in range(0, dim_nums)]
        # print(fake_data_train.shape)
        # print(data_train.shape)
        # update

    elif 10 <= dim_nums < 11:
        dim_flag = 1
        # reduction
        dims_cnt_pool = [0] * dim_nums
        print("starting dims reduction")
        random.seed(int(time.time()))
        dim_relation = [[0 for j in range(dim_nums)] for i in range(dim_nums)]
        for s in range(0, sample_nums):
            sample = data_train[s]
            sax = computeSax(sample)
            sax = sax.values
            # sax里面是series，要转成ndarray

            sax_list = []
            for v in range(0, sax.shape[0]):
                add = [0, 0, 0, 0]
                tag = sax[v][0].index
                res = list(sax[v][0].values)
                for i in range(0, len(tag)):
                    if tag[i] == 'a':
                        add[0] += res[i]
                    elif tag[i] == 'b':
                        add[1] += res[i]
                    elif tag[i] == 'c':
                        add[2] += res[i]
                    elif tag[i] == 'd':
                        add[3] += res[i]
                sax_list.append(add)
            # print(sax_list)
            for a in range(0, len(sax_list)):
                for b in range(a, len(sax_list)):
                    if a == b:
                        continue
                    sim = computeSim(sax_list[a], sax_list[b])
                    # print(sim)
                    if sim:
                        dim_relation[a][b] += 1
        dim_list = []
        for a in range(0, dim_nums):
            for b in range(a, dim_nums):
                if a == b:
                    continue
                dim_list.append([a, b, dim_relation[a][b]])
        # print(dim_list)
        dim_list.sort(key=lambda t: t[2], reverse=True)
        # 删减
        # TODO 这里只取了前一般的出现的概率
        dim_list = dim_list[:min(len(dim_list), int(len(dim_list) * 0.5) + 1)]
        # print(dim_list)
        cluster = {}
        for v in dim_list:
            if v[0] in cluster:
                cluster[v[0]].append(v[1])
            else:
                cluster[v[0]] = [v[1]]

        # 选取leader，通过list统计一下
        leader_list = []
        for i in cluster:
            leader_list.append(i)
            for j in cluster[i]:
                leader_list.append(j)
        counter = Counter(leader_list)
        # print(counter)
        # print(dim_list)
        # print(cluster)
        dim_choose = []
        check_dim_choose = [False] * dim_nums

        for k in cluster.keys():
            tmpl = [k]
            for a in cluster[k]:
                tmpl.append(a)
            choselst = []
            for a in tmpl:
                choselst.append([a, counter[a]])
            choselst.sort(key=lambda x: x[1], reverse=True)
            # print(choselst)
            pkidx = 0
            while True:
                if not check_dim_choose[choselst[pkidx][0]]:
                    dim_choose.append(choselst[pkidx][0])
                    break
                else:
                    pkidx += 1
                    if pkidx == len(choselst):
                        break
            check_dim_choose[k] = True
            for v in cluster[k]:
                check_dim_choose[v] = True
        for v in range(0, len(check_dim_choose)):
            if not check_dim_choose[v]:
                dim_choose.append(v)
        dim_choose = list(set(dim_choose))
        dim_choose.sort()

        # # sfa
        dim_relation = [[0 for j in range(dim_nums)] for i in range(dim_nums)]
        # print(dim_dic)
        for i in range(sample_nums):
            sample = data_train[i]
            a = sfa.SFA(n_jobs=-1)
            a.fit(sample)
            b = a.transform(sample, data_test)
            dim_dic = [collections.defaultdict(int) for i in range(dim_nums)]
            #
            for j in range(dim_nums):
                dt = b[0][j]
                for k in dt:
                    # print(k,dt[k])
                    dim_dic[j][k] += dt[k]

            dim_alphat_list = []
            for i in range(dim_nums):
                dim_col = dim_dic[i]
                # print(dim_col)
                tmplist = []
                for j in dim_col:
                    tmplist.append(j * dim_col[j])
                dim_alphat_list.append(Counter(''.join(tmplist)))
            # print(''.join(tmplist))
            # print(dim_alphat_list)

            for i in range(dim_nums):
                for j in range(i + 1, dim_nums):
                    s = dim_alphat_list[i]['a'] + dim_alphat_list[i]['b'] + dim_alphat_list[i]['c'] + \
                        dim_alphat_list[i]['d'] + \
                        dim_alphat_list[j]['a'] + dim_alphat_list[j]['b'] + dim_alphat_list[j]['c'] + \
                        dim_alphat_list[j]['d']
                    s //= 2
                    difa = abs(dim_alphat_list[i]['a'] - dim_alphat_list[j]['a'])
                    difb = abs(dim_alphat_list[i]['b'] - dim_alphat_list[j]['b'])
                    difc = abs(dim_alphat_list[i]['c'] - dim_alphat_list[j]['c'])
                    difd = abs(dim_alphat_list[i]['d'] - dim_alphat_list[j]['d'])
                    # print(i, j)
                    if (difa + difb + difc + difd) / s >= 0.4:
                        dim_relation[i][j] += 1

        dim_list = []
        for a in range(0, dim_nums):
            for b in range(a, dim_nums):
                if a == b:
                    continue
                dim_list.append([a, b, dim_relation[a][b]])
        # print(dim_list)
        dim_list.sort(key=lambda t: t[2], reverse=True)
        # print(dim_list)
        dim_list = dim_list[:min(len(dim_list), int(len(dim_list) * 0.5) + 1)]
        # print(dim_list)
        cluster = {}
        for v in dim_list:
            if v[0] in cluster:
                cluster[v[0]].append(v[1])
            else:
                cluster[v[0]] = [v[1]]

        # 选取leader，通过list统计一下
        leader_list = []
        for i in cluster:
            leader_list.append(i)
            for j in cluster[i]:
                leader_list.append(j)
        counter = Counter(leader_list)
        dim_choose2 = []
        check_dim_choose = [False] * dim_nums

        for k in cluster.keys():
            tmpl = [k]
            for a in cluster[k]:
                tmpl.append(a)
            choselst = []
            for a in tmpl:
                choselst.append([a, counter[a]])
            choselst.sort(key=lambda x: x[1], reverse=True)
            # print(choselst)
            pkidx = 0
            while True:
                if not check_dim_choose[choselst[pkidx][0]]:
                    dim_choose2.append(choselst[pkidx][0])
                    break
                else:
                    pkidx += 1
                    if pkidx == len(choselst):
                        break
            check_dim_choose[k] = True
            for v in cluster[k]:
                check_dim_choose[v] = True
        for v in range(0, len(check_dim_choose)):
            if not check_dim_choose[v]:
                dim_choose2.append(v)
        dim_choose2 = list(set(dim_choose2))
        dim_choose2.sort()
        dim_choose += dim_choose2
        dim_choose = list(set(dim_choose))
    else:
        dim_flag = 2
        dim_nums_chos = dim_nums // 5 + 1
        dim_choose = set([])
        for _ in range(dim_nums_chos):
            dim_pick = random.choice([i for i in range(dim_nums)])
            while dim_pick not in dim_choose:
                dim_choose.add(dim_pick)
            else:
                dim_pick = random.choices([i for i in range(dim_nums)])
        dim_choose = list(dim_choose)
        # print(dim_choose)
    if dim_flag != 0:
        # change dataset
        data_train = data_train.transpose((1, 0, 2))
        data_train_tmp = []
        for i in range(0, dim_nums):
            if i in dim_choose:
                data_train_tmp.append(data_train[i])
        data_train = np.array(data_train_tmp)
        data_train = data_train.transpose((1, 0, 2))

    old_to_new = {}
    dim_choose.sort()
    # print(dim_choose)
    for i in range(0, len(dim_choose)):
        old_to_new[i] = dim_choose[i]

    # correlation
    print("computing correlation...")
    dim_pool = correlation(data_train, old_to_new)
    fake_dim_pool = 0
    if dim_flag == 0:
        fake_dict = {}
        for i in range(0, len(fake_dim_choose)):
            fake_dict[i] = fake_dim_choose[i]
        fake_dim_pool = correlation(fake_data_train, fake_dict)
    # 用于输出相关矩阵
    # ans = isUseful(dim_pool)
    # print(ans)
    if times == 2:
        s = 'D://tmppro//FrameWork//test//1.txt'
        f = open(s, 'a')
        f.write(args.dataset + ',' + str(dim_pool) + ',' + '\n')
        f.close()

    dim_pool = np.array(dim_pool)
    print(dim_pool)
    print("correlation computed")
    # # judge_cor_matrix 维度*维度
    # # 该矩阵就是哪一维度相关就乘以1或0，权重参数设置为1/维度
    #
    #
    # # train
    print("training...")
    sc = 0
    if dim_flag != 0:
        clf = _mycif.NewCanonicalIntervalForest(dim_pool=dim_pool, n_jobs=-1, n_intervals=500)
        clf.fit(res_data_train, target_train)
        _mycif.NewCanonicalIntervalForest(...)
        y_pred = clf.predict(data_test)
        sc = accuracy_score(target_test, y_pred)
        print("trained,accuracy is:")
        print(sc)
        print("saving score")
        # path to save
    else:
        clf = _mycif.NewCanonicalIntervalForest(dim_pool=dim_pool, n_jobs=-1, n_estimators=500)
        clf.fit(res_data_train, target_train)
        pre1 = clf.predict_proba(data_test)
        clf = _mycif.NewCanonicalIntervalForest(dim_pool=fake_dim_pool, n_jobs=-1, n_estimators=500)
        clf.fit(fake_data_train, target_train)
        fake_data_test = []
        for s in range(0, data_test.shape[0]):
            sample = data_test[s]
            x = []
            for sm in sample:
                length = len(sm)
                # print(sm)
                dif = [0 for i in range(length)]
                for i in range(1, length):
                    dif[i - 1] = sm[i] - sm[i - 1]
                dif[-1] = dif[-2]
                dif = np.array(dif)
                # print(sm,dif)
                x.append(dif)
            # print(x.shape)
            fake_data_test.append(np.array(x))
            # print("---------------")
        fake_data_test = np.array(fake_data_test)
        pre2 = clf.predict_proba(fake_data_test)
        pre = pre1 + pre2
        lb = np.asarray([np.argmax(prob) for prob in pre])
        sc = accuracy_score(target_test, lb)
        print(sc)
    #
    s = 'D://tmppro//FrameWork//test//' + str(times) + '.csv'
    # s = 'D://tmppro//cif_tmp//' + '0' + '.csv'
    f = open(s, 'a')
    f.write(args.dataset + ',' + str(sc) + ',' + '\n')
    f.close()
    print("score saved")


for i in range(0, args.times):
    run(i)
