# ! PyCharm
# -*- Created by panyue  -*-
import Logistic
import numpy as np
import matplotlib.pyplot as plt

"""
func ：加载数据
Parameters : None
Returns : dataMat - 数据列表
          labelMat - 标签列表
"""


def loadDataSet():
    # 创建数据列表
    dataMat = []
    # 创建标签列表
    labelMat = []
    # 打开文件
    fr = open('testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # 去掉每行两边的空白字符，并以空格分隔每行数据元素
        lineArr = line.strip().split()
        # 添加数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(int(lineArr[2]))
    # 关闭文件
    fr.close()
    # 返回
    return dataMat, labelMat


"""
func ：绘制数据集
Parameters :  weights - 权重参数数组 
Returns : None
"""


def plotBestFit(weights):
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    # 转换成numpy的array数组
    dataArr = np.array(dataMat)
    # 数据个数
    # 例如建立一个4*2的矩阵c，c.shape[1]为第一维的长度2， c.shape[0]为第二维的长度4
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        if int(labelMat[i]) == 1:
            # 1为正样本
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            # 0为负样本
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 新建图框
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    # x轴坐标
    x = np.arange(-3.0, 3.0, 0.1)
    # w0*x0 + w1*x1 * w2*x2 = 0
    # x0 = 1, x1 = x, x2 = y
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    # 绘制title
    plt.title('BestFit')
    # 绘制label
    plt.xlabel('x1')
    plt.ylabel('y2')
    # 显示
    plt.show()


def main():
    dataMat, labelMat = loadDataSet()
    # 梯度上升法
    weights2, weights_array2 = Logistic.gradAscent(dataMat, labelMat)
    plotBestFit(weights2)
    print(Logistic.gradAscent(dataMat, labelMat))

    # 随机梯度上升法
    weights0 = Logistic.stocGradAscent0(np.array(dataMat), labelMat)
    plotBestFit(weights0)

    # 改进的随机梯度上升法
    weights1, weights_array1 = Logistic.stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights1)
    print(Logistic.stocGradAscent1(dataMat, labelMat))

    Logistic.plotWeights(weights_array1, weights_array2)


if __name__ == '__main__':
    main()
