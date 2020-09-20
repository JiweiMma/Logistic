import numpy as np
import random

#sigmoid函数
def sigmoid(inX):
    if inX >= 0:
        return 1.0 / (1 + np.exp(-inX))
    else:
        return np.exp(inX) / (1 + np.exp(inX))

#函数说明:改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小，m为行数,n为列数。
    m,n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            # 选择随机选取的一个样本，计算h
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除已经使用的样本
            del(dataIndex[randIndex])
    return weights


#使用Python写的Logistic分类器做预测
def colicTest():
    #打开训练集
    frTrain = open('horseColicTraining.txt')
    #打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随机上升梯度训练
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        #计算测试总数
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    #计算错误率
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

#分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
    colicTest()