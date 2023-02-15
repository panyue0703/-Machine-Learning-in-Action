# -*- coding: utf-8 -*-

import numpy as np
import random

"""
func ：sigmoid函数
Parameters : inX - 数据
Returns : sigmoid函数

"""


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
func ：梯度上升法
Parameters : dataMath - 数据集
             classLabels - 数据标签
Returns : weights.getA() - 求得的权重数组（最优参数）
          weights_array - 每次更新的回归系数
"""


def gradAscent(dataMatIn, classLabels):
    # 转换为Numpy矩阵数据类型
    # dataMatIn为100✖️3矩阵（X0,X1,X2）
    dataMatrix = np.mat(dataMatIn)  # 2维Numpy数组转为矩阵
    labelMat = np.mat(classLabels).transpose()  # 转置，行向量转为列向量
    m, n = np.shape(dataMatrix)  # m=100,n=3
    # 梯度上升算法所需参数
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数

    weights = np.ones((n, 1))  # 3✖️1
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix * weights)  # 100✖️1 列向量
        error = (labelMat - h)  # 计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  # 3X100✖️100X1=3✖️1
    return weights

"""
func ：随机梯度上升法
Parameters : dataMatrix - 数据数组
             classLabels - 数据标签

Returns :  weights - 求得的回归系数数组（最优参数）
"""


def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


"""
func ：改进的随机梯度上升法
Parameters : dataMatrix - 数据数组
             classLabels - 数据标签
             numIter - 迭代次数
Returns : weights - 求得的回归系数数组（最优参数）
"""


def stocGradAscent1(dataMatrix, classLabels, numIter):
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次都降低alpha的大小
            alpha = 4 / (1.0 + j + i) + 0.001
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选择一个样本计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除已使用的样本
            del (dataIndex[randIndex])
    # 返回
    return weights

"""
func ：用python写的Logistic分类器做预测
Parameters : None
Returns : None
"""


def colicTest():
    # 打开训练集
    frTrain = open('horseColicTraining.txt')
    # 打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # trainingLabels.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用随机梯度上升
    # trainWeights = stocGradAscent0(np.array(trainingSet), trainingLabels)
    # 使用改进的随机上升梯度训练
    # trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    # 使用梯度上升训练
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        # if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
        if int(classifyVector(np.array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    # 错误概率计算
    errorRate = (float(errorCount) / numTestVect) * 100
    print("the error rate of this test is: %.2f%%" % errorRate)


"""
func ：分类函数
Parameters : inX - 特征向量
             weights - 回归系数
Returns : 分类结果
"""


def classifyVector(inX, weights):
    prob =sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    colicTest()
