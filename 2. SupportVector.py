import csv
import numpy as np
import math
from sklearn.model_selection import train_test_split


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
    _result = (_class_1 * _class_2) * _matrixItem + (math.pow(_lambda, 2))

    return _result

def MeasuringAlpha(_matriksHessian, _alpha, _gamma, C):
    
    # Mencari nilai error data ke-i
    # Alpha di inisialisasi = 0
    # Lakulan perulangan hingga didapatkan semua nilai
    # Di paper Alpha yang didapatkan = 0.5
    # Output dari nilai error berupa vektor sebanyak tanggapan
    
    _listOfError = []
    _listOfNewAlpha = []
    _listofAlpha = []
    _deltaAlpha = []

    tempError = 0
    tempDelta = 0
    _newListOfTempError = []

    print("Ini Hessian Matrix :", _matriksHessian)

    for _ in range(len(_matriksHessian)):
        _listofAlpha.append(0)

    for i in range(0, 10):
        for _, row in enumerate(_matriksHessian): # Mencari Ei
            for j, var in enumerate(row):
                tempError = tempError + (_listofAlpha[j] * var)

        for j, itemError in enumerate(_newListOfTempError): # Mencari delta alpha
            if not _listOfNewAlpha:
                tempDelta = (min(max( (_gamma * (1 - itemError)), - 0), C - 0))
            else:
                tempDelta = (min(max( (_gamma * (1 - itemError)), -_listOfNewAlpha[j]), C - _listOfNewAlpha[j]))

        if (len(_deltaAlpha) < len(_newListOfTempError)):
            _deltaAlpha.append(tempDelta)
        else:
            _deltaAlpha[j] = tempDelta

        if  not _listOfNewAlpha:
            _alpha = [iAlpha + iDelta for iAlpha, iDelta in zip(_listOfNewAlpha, _deltaAlpha)]
        else:
            for k in range(len(_deltaAlpha)):
                tempX = _alpha + _deltaAlpha[k]
                _listOfNewAlpha.append(tempX)
    
    print("New list error: ", _newListOfTempError)
    print("Delta Alpha: ", _deltaAlpha)
    print("New Alpha: ", _listOfNewAlpha)
                

                





def LinearKernel(_xTrain, y_train, _lambda):
    
    newValueOfMatrix = []
    _xTrainTranpose = _xTrain.transpose()    
    matrixGenerate = _xTrain.dot(_xTrainTranpose)
    
    for i in range(len(matrixGenerate)):
        tempList = []
        for j, var in enumerate(matrixGenerate[i]):
            var = HessianMatrixFunction(y_train[i], y_train[j], var, _lambda)
            # print('HessianMatrixFunction(%d, %d, %d, 0.5)' % (y_train[i], y_train[j], var))
            tempList.append(var)
        newValueOfMatrix.insert(i, tempList)

    newValueOfMatrix = np.array(newValueOfMatrix)

    return newValueOfMatrix

def main():
    # Inisialisasi 
    filename = 'Result/TF_IDF.csv'
    splitting = 0.1

    mainAlpha = 0
    setGamma = 0.5
    setLambda = 0.5
    setEpisolon = 0.001
    C = 1


    vectorOfOpinion, valueOfSentiment = ReadCSV(filename)
    x_train, x_test, y_train, y_test = train_test_split(vectorOfOpinion, valueOfSentiment, test_size=splitting, random_state=0)
    hessianMatrix = LinearKernel(x_train, y_train, setLambda)
    # hessianMatrix = LinearKernel(vectorOfOpinion, valueOfSentiment)
    MeasuringAlpha(hessianMatrix, mainAlpha, setGamma, C)
    # print(hessianMatrix.shape)
    # print(valueOfSentiment)

    # print(hessianMatrix)

main()