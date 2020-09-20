from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

#加载数据
def loadDataSet():
    #创建数据列表
    dataMat = []
    #创建标签列表
    labelMat = []
    #打开文件
    fr = open('testSet.txt')
    #读取文件
    for line in fr.readlines():
        #去掉空格回车，放入列表
        lineArr = line.strip().split()
        #添加数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #添加标签
        labelMat.append(int(lineArr[2]))
    #关闭文件
    fr.close()
    #返回标签和数据
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


#函数说明:绘制数据集
def plotBestFit(weights):
    #加载数据
    dataMat, labelMat = loadDataSet()
    #转换成numpy的array数组
    dataArr = np.array(dataMat)
    #计算数据个数
    n = np.shape(dataMat)[0]
    #正样本字典
    xcord1 = []; ycord1 = []
    #负样本字典
    xcord2 = []; ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        # 1为正样本
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        # 0为负样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #绘制样本
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    #weights - 权重参数数组
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


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

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)