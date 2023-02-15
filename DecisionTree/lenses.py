# ! PyCharm
# -*- Created by panyue  -*-
import tree
import treePlotter

with open('lenses.txt') as fr:
    # 处理文件，去掉每行两头的空白符，以\t分隔每个数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_targt = []
for each in lenses:
    # 存储Label到lenses_targt中
    lenses_targt.append([each[-1]])
# 特征标签
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

lensesTree = tree.createTree_C45(lenses, lensesLabels)

treePlotter.createPlot(lensesTree)
