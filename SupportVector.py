import csv
import numpy as np
import math

def ReadCSV(_filename):
    
    sentimentList = []
    valueOfSentiment = []

    firstline = True
    with open(_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if firstline:
                firstline = False
                continue
            # print(row[1:-1])
            sentimentList.append(row[1:-1])
            valueOfSentiment.append(int(row[-1]))
    
    for i in range(len(sentimentList)):
        for j in range(len(sentimentList[i])):
            sentimentList[i][j] = float(sentimentList[i][j])
    
    sentimentList = np.array(sentimentList)

    return sentimentList, valueOfSentiment

def HessianMatrixFunction(_class_1, _class_2, _matrixItem, _lambda):
    result = (_class_1 * _class_2) * _matrixItem + (math.pow(_lambda, 2))

    return result

def KernelRBF(_sentimentList, _valueOfSentiment):
    
    newValueOfMatrix = []
    transposeOfSentiment = _sentimentList.transpose()    
    matrixGenerate = _sentimentList.dot(transposeOfSentiment)
    
    for i in range(len(matrixGenerate)):
        tempList = []
        for j, var in enumerate(matrixGenerate[i]):
            var = HessianMatrixFunction(_valueOfSentiment[i], _valueOfSentiment[j], var, 0.5)
            tempList.append(var)
        newValueOfMatrix.insert(i, tempList)

    newValueOfMatrix = np.array(newValueOfMatrix)

    return newValueOfMatrix

def main():
    
    vectorOfSentiment, valueOfSentiment = ReadCSV('Result/TF_IDF.csv')
    hessianMatrix = KernelRBF(vectorOfSentiment, valueOfSentiment)
    # print(valueOfSentiment)

    print(hessianMatrix)

main()