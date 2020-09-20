import numpy as np

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

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print(gradAscent(dataMat, labelMat))
