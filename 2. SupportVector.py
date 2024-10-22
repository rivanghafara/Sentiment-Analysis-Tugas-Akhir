import csv
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from sklearn import svm

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

# def HessianMatrixFunction(_class_1, _class_2, _matrixItem, _lambda):
#     result = (_class_1 * _class_2) * _matrixItem + (math.pow(_lambda, 2))

#     return result

# def KernelRBF(_valueOfSentiment, _gamma):
    
   

def main():

    vectorOfOpinion, valueOfSentiment = ReadCSV('Result/TF_IDF.csv')
    # print(vectorOfOpinion.shape)
    xTrain, xTest, yTrain, yTest = train_test_split(vectorOfOpinion, valueOfSentiment, test_size=0.1, random_state=0)

    # print(xTrain.shape, xTest.shape)
    # print("="*100)
    # print(yTrain.shape, yTest.shape)

    clf = svm.SVC(kernel='linear', C=1).fit(xTrain, yTrain)
    scores = cross_val_score(clf, xTrain, yTrain, cv=10)
    # print(iris.data)
    # print("="*100)
    # print(iris.target)
    # print(xTrain)
    # print("="*100)
    # print(xTest)
    # print("="*100)
    # print(yTrain)
    # print("="*100)
    # print(yTest)
    # print(valueOfSentiment[89:])

    print(clf.score(xTest, yTest))
    print(scores)

main()