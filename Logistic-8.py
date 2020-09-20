from sklearn.linear_model import LogisticRegression

#函数说明:使用Sklearn构建Logistic回归分类器
def colicSklearn():
    #打开训练集
    frTrain = open('horseColicTraining.txt')
    #打开测试集
    frTest = open('horseColicTest.txt')
    #创建原始的训练集测试集标签和数据字典
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        #去掉空格回车等
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='lbfgs',max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
    colicSklearn()