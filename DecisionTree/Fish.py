# ! PyCharm
# -*- Created by panyue  -*-
from matplotlib.font_manager import FontProperties

import matplotlib
import tree
import treePlotter

"""
func：创建测试数据集
Parameters ：None
Returns : dataSet - 数据集
         labels - 分类属性
"""

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


"""

gunc：main函数

Parameters:None

Returns None

"""


def main():
    myDat, labels1 = createDataSet()
    labels2 = labels1[:]
    # myDat[0][-1] = 'maybe'
    print(myDat)
    print('labels1:', labels1)
    print('Shannon:', tree.calcShannonEnt(myDat))

    # print(splitDataSet(myDat,2,1))
    # print(chooseBestFeatureToSplit(myDat))
    FishTree = tree.createTree_ID3(myDat, labels1)
    print('labels2:', labels2)
    print(FishTree)
    print('=' * 88)
    testVec = [0, 1, 0]
    treePlotter.createPlot(FishTree)
    print('labels2:', labels2)

    label = tree.classify(FishTree, labels2, testVec)
    print('testVec:{}-->{}'.format(testVec, label))
    print('*' * 88)
    print(FishTree)
    tree.storeTree(FishTree, 'FishTree.txt')


if __name__ == '__main__':
    main()

