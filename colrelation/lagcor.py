import math

import numpy as np


def lag_cor(s1, s2, t, w, l):
    """
    计算窗口处的滞后相似性
    :param s1: 序列1
    :param s2: 序列2
    :param t: 时间点
    :param w: 窗口大小
    :param l: 滞后长度
    :return: 滞后相似性
    """
    if l < 0:
        s1, s2 = s2, s1
        l *= -1
    # st = t - w + 1
    # ed = t - l
    # # print(st,ed)
    # up = 0
    # seq1 = s1[t - (w - l) + 1:t + 1]
    # seq2 = s2[t - l - (w - l) + 1:t - l + 1]
    # for i in range(st, ed + 1):
    #     avg1 = np.average(seq1)
    #     avg2 = np.average(seq2)
    #     up += (s1[i + l] - avg1) * (s2[i] - avg2)
    # down = np.std(seq1, ddof=1) * np.std(seq2, ddof=1)
    # return up / down
    seq1 = s1[t - (w - l) + 1:t + 1]
    seq2 = s2[t - l - (w - l) + 1:t - l + 1]
    if len(seq1) != len(seq2):
        return 0
    if len(seq1) == 1:
        return 0
    innerdot = np.dot(seq1, seq2)
    sum1 = np.sum(np.array(seq1))
    sum2 = np.sum(np.array(seq2))
    sum3 = np.sum(np.array(seq1) * np.array(seq1))
    sum4 = np.sum(np.array(seq2) * np.array(seq2))
    tmp1 = sum3 - sum1 * sum1 / (w - l)
    tmp2 = sum4 - sum2 * sum2 / (w - l)
    tmp1 = max(tmp1, 0)
    tmp2 = max(tmp2, 0)
    fangcha1 = math.sqrt(tmp1)
    fangcha2 = math.sqrt(tmp2)
    down = fangcha2 * fangcha1
    up = innerdot - (sum1 * sum2) / (w - l)
    if down == 0:
        return 1
    return up / down


def aggregated(s1, s2, t, w, l):
    """
    两个时间序列之间的聚合滞后相关性
    :param s1: 序列1
    :param s2: 序列2
    :param t: 时间点
    :param w: 窗口大小
    :param l: 滞后长度
    :return: 聚合滞后相关性,1代表第一条序列lead第二条序列，2代表第二条序列lead第一条序列
    """
    maxlag = w // 2
    max_pos_cor = 0
    min_pos_cor = 0
    for i in range(0, maxlag + 1):
        max_pos_cor += max(0, lag_cor(s1, s2, t, w, i))
    for i in range(-maxlag, 0):
        min_pos_cor += max(0, lag_cor(s1, s2, t, w, i))

    max_pos_cor /= maxlag + 1
    min_pos_cor /= maxlag
    if max_pos_cor >= min_pos_cor:
        return max_pos_cor, 2
    return min_pos_cor, 1


# 图的节点结构
class Node:
    def __init__(self, value):
        self.value = value  # 节点值
        self.come = 0  # 节点入度
        self.out = 0  # 节点出度
        self.nexts = []  # 节点的邻居节点
        self.edges = []  # 在节点为from的情况下，边的集合


# 图的边结构
class Edge:
    def __init__(self, weight, fro, to):
        self.weight = weight  # 边的权重
        self.fro = fro  # 边的from节点
        self.to = to  # 边的to节点


# 图结构
class Graph:
    def __init__(self):
        self.nodes = {}  # 图的所有节点集合  字典形式：{节点编号：节点}
        self.edges = []  # 图的边集合


# 生成图结构
# matrix = [
#   [1,2,3],        ==>   里面分别代表权重, from节点, to节点
#   [...]
# ]
def createGraph(matrix):
    graph = Graph()
    for edge in matrix:
        weight = edge[0]
        fro = edge[1]
        to = edge[2]
        if fro not in graph.nodes:
            graph.nodes[fro] = Node(fro)
        if to not in graph.nodes:
            graph.nodes[to] = Node(to)
        fromNode = graph.nodes[fro]
        toNode = graph.nodes[to]
        newEdge = Edge(weight, fromNode, toNode)
        fromNode.nexts.append(toNode)
        fromNode.out += 1
        toNode.come += 1
        fromNode.edges.append(newEdge)
        graph.edges.append(newEdge)
    return graph
