import numpy as np
from math import sqrt
from collections import Counter

class kNNClassifier:
    # 定义构造函数
    def __init__(self, k):

        assert k>=1, "无效的K值"
        self.k = k
        # 加上_定义为protected变量
        self._xTrain = None
        self._yTrain = None

    # 定义fit方法
    def fit(self, xTrain, yTrain):

        self._xTrain = xTrain
        self._yTrain = yTrain
        return self

    # 定义predict方法
    def predict(self, xPredict):

        assert self._xTrain is not None and self._yTrain is not None, "请先进行Fit"
        assert xPredict.shape[1] == self._xTrain.shape[1], "特征数量与训练数据集不一致（列数不一致）"
        
        yPredict = [self._predict(x) for x in xPredict]
        return np.array(yPredict)

    def _predict(self, x):

        assert x.shape[0] == self._xTrain.shape[1], "特征数量与训练数据集不一致（列数不一致）"

        distances = [sqrt(np.sum((xTrain - x)**2)) for xTrain in self._xTrain]
        nearest = np.argsort(distances)
        topKY = [self._yTrain[i] for i in nearest[:self.k]]
        votes = Counter(topKY)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "kNN(k=%d)" % self.k
