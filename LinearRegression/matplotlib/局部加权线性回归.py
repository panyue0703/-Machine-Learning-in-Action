#coding: utf-8
from numpy import *

#==================�ֲ���Ȩ���Իع�================
# �������ݼ�
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))   #�����Խ��߾���
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        #����Ȩ��ֵ����ָ�����ݼ�
        weights[j,j] = exp(diffMat * diffMat.T /(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "this matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] =lwlr(testArr[i],xArr,yArr,k)
    return yHat


xArr,yArr = loadDataSet('ex0.txt')
print "k=1.0��",lwlr(xArr[0],xArr,yArr,1.0) # Ƿ���
print "k=0.01��",lwlr(xArr[0],xArr,yArr,0.001)
print "k=0.003��",lwlr(xArr[0],xArr,yArr,0.003) # �����

#��ͼ
def showlwlr():
    yHat = lwlrTest(xArr, xArr, yArr, 0.01) #�Ĳ�
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    import matplotlib.pyplot as plt
    fig = plt.figure() #������ͼ����
    ax = fig.add_subplot(111)  #111��ʾ����������Ϊ1��2��ѡ��ʹ�ô��ϵ��µ�һ��
    ax.plot(xSort[:,1],yHat[srtInd])
    #scatter����ɢ��ͼ
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T[:,0].flatten().A[0],s=2,c='red')
    plt.show()

showlwlr()