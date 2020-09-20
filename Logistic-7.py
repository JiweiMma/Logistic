import numpy as np
import random

#sigmoid函数
def sigmoid(in_X):
    #return 1.0 / (1 + np.exp(-in_X))
    return .5 * (1 + np.tanh(.5 * in_X))

#梯度上升算法
#weights.getA() - 求得的权重数组(最优参数)
#dataMatIn - 数据集 classLabels - 数据标签
def gradAscent(dataMatIn, classLabels):
    # 转换成numpy的mat
    dataMatrix = np.mat(dataMatIn)
    # 转换成numpy的mat,并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小。m为行数,n为列数
    m, n = np.shape(dataMatrix)
    #步长,学习速率,控制更新的幅度。
    alpha = 0.001
    #最大的迭代次数
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    return weights.getA()

#使用Python写的Logistic分类器做预测
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    #使用梯度上升算法训练
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0]))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

#分类函数
def classifyVector(in_X, weights):
    prob = sigmoid(sum(in_X*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
    colicTest()