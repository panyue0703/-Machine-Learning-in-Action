# ! PyCharm
# -*- Created by panyue  -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib
from math import log
import operator
import pickle

matplotlib.use('TkAgg')

"""

func ：计算给定数据集的经验熵（香农熵）

Parameters : dataSet - 数据集

Returns : shannonEnt - 经验熵（香农熵）

"""


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数

    # 保存每个标签（Label）出现次数的“字典”
    labelCounts = {}

    # 对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签（Label）信息

        if currentLabel not in labelCounts.keys():  # 如果标签（Label）没有放入统计次数的字典，添加进去
            # 创建一个新的键值对，键为currentLabel值为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    # 经验熵（香农熵）
    shannonEnt = 0.0

    # 计算香农熵
    for key in labelCounts:
        # 选择该标签（Label）的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    # 返回经验熵（香农熵）
    return shannonEnt


"""

func：按照给定特征划分数据集

Parameters : dataSet - 待划分的数据集
             axis - 划分数据集的特征
             values - 需要返回的特征的值

Returns : None 

"""


def splitDataSet(dataSet, axis, value):
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集的每一行
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            reducedFeatVec.extend(featVec[axis + 1:])
            # 列表中嵌套列表
            retDataSet.append(reducedFeatVec)
            # 返回划分后的数据集
    return retDataSet


"""

func ：选择最优特征 ID3

Parameters : dataSet - 数据集

Returns : bestFeature - 信息增益最大的（最优）特征的索引值

"""


def chooseBestFeatureToSplit_ID3(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算原始香农熵，保存最初的无序度量值
    baseEntropy = calcShannonEnt(dataSet)

    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1

    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0

        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 计算信息增益
        if infoGain > bestInfoGain:
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i

    # 返回信息增益最大的特征的索引值
    return bestFeature


"""

func ：选择最优特征 C4.5
Parameters : dataSet - 数据集
Returns : bestFeature - 信息增益最大的（最优）特征的索引值

"""


def chooseBestFeatureToSplit_C45(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain_ratio = 0.0
    # 最优特征的索引
    bestFeature = -1

    for i in range(numFeatures):  # check every feature
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        IV = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # C4.5
            IV = IV - prob * log(prob, 2)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        if IV == 0:
            continue
        infoGain_ratio = infoGain / IV  # infoGain_ratio of current feature

        if infoGain_ratio > bestInfoGain_ratio:  # choose the greatest gain ratio
            bestInfoGain_ratio = infoGain_ratio
            # 记录信息增益最大的特征的索引值
            bestFeature = i

    return bestFeature


"""

func ：选择最优特征 CART
Parameters : dataSet - 数据集
Returns : bestFeature - 信息增益最大的（最优）特征的索引值

"""


def chooseBestFeature_CART(dataSet):
    numFeatures = len(dataSet[0]) - 1  # except the column of labels
    bestGini = 999999.0
    bestFeature = -1  # default label

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # get the possible values of each feature
        gini = 0.0

        for value in uniqueVals:
            subdataset = splitDataSet(dataSet, i, value)
            prob = len(subdataset) / float(len(dataSet))
            subp = len(splitDataSet(subdataset, -1, '0')) / float(len(subdataset))
        gini += prob * (1.0 - pow(subp, 2) - pow(1 - subp, 2))

        if gini < bestGini:
            bestGini = gini
            bestFeature = i

    return bestFeature


"""
func：统计classList中出现次数最多的元素（类标签）
        服务于递归第两个终止条件
Parameters : classList - 类标签列表
Returns : sortedClassCount[0][0] - 出现次数最多的元素（类标签）

"""


def majorityCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排序
    # operator.itemgetter(1)获取对象的第1列的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]


"""

func：创建决策树（ID3算法）
Parameters : dataSet - 训练数据集
             labels - 分类属性标签
             featLabels - 存储选择的最优特征标签
Returns : myTree - 决策树

"""


def createTree_ID3(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit_ID3(dataSet)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + bestFeatLabel)
    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    # print(featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = createTree_ID3(splitDataSet(dataSet, bestFeat, value), subLabels)

    return ID3Tree


"""

func：创建决策树（C4.5算法）
Parameters : dataSet - 训练数据集
             labels - 分类属性标签
Returns : myTree - 决策树

"""


def createTree_C45(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # classList[0]元素的个数，与列表长度相等，即类别完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit_C45(dataSet)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 从属性列表中删掉已经被选出来当根节点的属性
    featValues = [example[bestFeat] for example in dataSet]
    # print(featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree_C45(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


"""

func：创建决策树（CART算法）
Parameters : dataSet - 训练数据集
             labels - 分类属性标签
Returns : myTree - 决策树

"""


def createTree_CART(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # print('-'*88)
    # print(classList)
    # classList[0]元素的个数，与列表长度相等，即类别完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeature_CART(dataSet)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    # print(featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree_CART(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


"""
func：使用决策树分类

Parameters:inputTree - 已经生成的决策树
           featLabels - 存储选择的最优特征标签
           testVec - 测试数据列表，顺序对应最优特征标签

Returns:classLabel - 分类结果

"""


def classify(inputTree, featLabels, testVec):
    # 获取决策树结点
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    # firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


"""
构造决策树会很耗时，数据集很大时，会耗费大量时间，但是若执行分类时，运用提前训练好的决策树，则会节省很多计算时间
python模块pickle，可以序列化对象，将训练好的决策树保存在磁盘上，并在需要时读取出来

func：存储决策树

Parameters:inputTree - 已经生成的决策树
           filename - 决策树的存储文件名

Returns:None

"""


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


"""
func：读取决策树

Parameters:filename - 决策树的存储文件名

Returns: pickle.load(fr) - 决策树字典

"""


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


# pre_pruning = False
pre_pruning = True
# post_pruning = True
post_pruning = False
from collections import Counter


# 利用ID3算法创建决策树
def ID3_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit_ID3(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))

    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    # if pre_pruning:
    #     ans = []
    #     for index in range(len(test_dataset)):
    #         ans.append(test_dataset[index][-1])
    #     result_counter = Counter()
    #     for vec in dataset:
    #         result_counter[vec[-1]] += 1
    #     leaf_output = result_counter.most_common(1)[0][0]
    #     root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)
    #     outputs = []
    #     ans = []
    #     for value in uniqueVals:
    #         cut_testset = splitDataSet(test_dataset, bestFeat, value)
    #         cut_dataset = splitDataSet(dataset, bestFeat, value)
    #         for vec in cut_testset:
    #             ans.append(vec[-1])
    #         result_counter = Counter()
    #         for vec in cut_dataset:
    #             result_counter[vec[-1]] += 1
    #         leaf_output = result_counter.most_common(1)[0][0]
    #         outputs += [leaf_output] * len(cut_testset)
    #     cut_acc = cal_acc(test_output=outputs, label=ans)
    #
    #     if cut_acc <= root_acc:
    #         return leaf_output
    #
    # for value in uniqueVals:
    #     subLabels = labels[:]
    #     ID3Tree[bestFeatLabel][value] = ID3_createTree(
    #         splitDataSet(dataset, bestFeat, value),
    #         subLabels,
    #         splitDataSet(test_dataset, bestFeat, value))
    #
    # if post_pruning:
    #     tree_output = classify(ID3Tree,
    #                            featLabels=['年龄段', '有工作', '有自己的房子', '信贷情况'])
    #     ans = []
    #     for vec in test_dataset:
    #         ans.append(vec[-1])
    #     root_acc = cal_acc(tree_output, ans)
    #     result_counter = Counter()
    #     for vec in dataset:
    #         result_counter[vec[-1]] += 1
    #     leaf_output = result_counter.most_common(1)[0][0]
    #     cut_acc = cal_acc([leaf_output] * len(test_dataset), ans)
    #
    #     if cut_acc >= root_acc:
    #         return leaf_output

    return ID3Tree


# def cal_acc(test_output, label):
#     """
#     :param test_output: the output of testset
#     :param label: the answer
#     :return: the acc of
#     """
#     assert len(test_output) == len(label)
#     count = 0
#     for index in range(len(test_output)):
#         if test_output[index] == label[index]:
#             count += 1
#
#     return float(count / len(test_output))
