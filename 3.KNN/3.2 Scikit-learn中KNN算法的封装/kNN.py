import numpy as np
from math import sqrt
from collections import Counter

'''
k：kNN中的k，判断多少个最近的数据
xTrain：待训练的特征数据
yTrain：待训练的Label标记数据
x：新的待预测判断的数据
return 返回
'''
def kNNClassify(k, xTrain, yTrain, x):

    # k需大于1且小于训练集样本数量
    print(k)
    print(xTrain.shape[0])
    if 1 <= k <= xTrain.shape[0]:
        print("无效的K值")
        return
    # 训练数据特征需与Label一一对应数量相同
    if xTrain.shape[0] != yTrain.shape[0]:
        print("训练数据特征与Label数量不同")
        return
    # 训练数据特征维度需与待预测的数据相同
    if xTrain.shape[0] != x.shape[0]:
        print("训练数据特征维度与待预测的数据不同")
        return

    distances = [sqrt(np.sum((xTrain - x)**2)) for xTrain in xTrain]
    nearest = np.argsort(distances)
    topKY = [yTrain[i] for i in nearest[:k]]
    votes = Counter(topKY)

    return votes.most_common(1)[0][0]