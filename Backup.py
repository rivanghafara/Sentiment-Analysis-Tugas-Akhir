import csv
import numpy as np
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm


def ReadCSV(_filename):
    
    idSentiment = []
    sentimentList = []
    valueOfSentimentAsrama = []
    valueOfSentimentKesehatan = []
    valueOfSentimentAsuransi = []
    valueOfSentimentBeasiswa = []
    valueOfSentimentKegiatanMahasiswa = []

    firstline = True
    with open(_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if firstline:
                firstline = False
                continue
            # print(row[1:-1])
            idSentiment.append(int(row[0]))
            sentimentList.append(row[1:-6])
            valueOfSentimentAsrama.append(int(row[-5]))
            valueOfSentimentKesehatan.append(int(row[-4]))
            valueOfSentimentAsuransi.append(int(row[-3]))
            valueOfSentimentBeasiswa.append(int(row[-2]))
            valueOfSentimentKegiatanMahasiswa.append(int(row[-1]))
    
    for i in range(len(sentimentList)):
        for j in range(len(sentimentList[i])):
            sentimentList[i][j] = float(sentimentList[i][j])
    
    sentimentList = np.array(sentimentList)

    return idSentiment, sentimentList, valueOfSentimentAsrama, valueOfSentimentKesehatan, valueOfSentimentAsuransi, valueOfSentimentBeasiswa, valueOfSentimentKegiatanMahasiswa

def WriteToCSV(_filename, _listing):
    header = 'Index', 'Asrama', 'Kesehatan', 'Asuransi', 'Beasiswa', 'Kegiatan Mahasiswa'
    with open(_filename, 'w', newline='') as file:
        reswrite = csv.writer(file, delimiter=',')
        reswrite.writerow(header)
        for i, var in enumerate(_listing):
            var = list(var)
            var.insert(0, 'C = %s' %(i))
            reswrite.writerow(var)

def SVM_Classification(_vectorOfOpinion, _vectorOfSentiment, _testSize, _nameOfOpinion, _C):

    xTrain, xTest, yTrain, yTest = train_test_split(_vectorOfOpinion, _vectorOfSentiment, test_size=_testSize, random_state=1)
    clf = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrain)

    predicted = clf.predict(xTest)
    accuracy = accuracy_score(yTest, predicted)*100
    
    print("Nilai akurasi %s =" %(_nameOfOpinion), accuracy)
    

    return accuracy

def SVM_Classification_With_CV(_idSentiment, _vectorOfOpinion, _vectorOfSentiment, _nameOfOpinion, _C, numFold):
    # xTrain, xTest, yTrain, yTest = train_test_split(_vectorOfOpinion, _vectorOfSentiment, test_size=_testSize, random_state=1)

    # accuracy = accuracy_score(yTest, predicted)*100
    # print("panjang vecotr : ", _vectorOfOpinion[5])
    subset_size = int(len(_idSentiment)/numFold)
    tempSentiment = []

    for i in range(numFold):
        print("================= Iterasi ",i,"=====================")
        idTest = _idSentiment[i * subset_size:][:subset_size]
        idTrain =  _idSentiment[:i * subset_size] + _idSentiment[(i+1)*subset_size:]
        
        xTrain = []
        xTest = []

        yTrain = []
        yTest = []

        #ambil data training
        for j in range(len(idTrain)):
            xTrain.append(_vectorOfOpinion[idTrain[j]])
            yTrain.append(_vectorOfSentiment[idTrain[j]])

        #ambil data tsesting
        for j in range(len(idTest)):
            xTest.append(_vectorOfOpinion[idTest[j]])
            yTest.append(_vectorOfSentiment[idTest[j]])

        clf = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrain)

        predicted = clf.predict(xTest)
        for a, b in zip(idTest, predicted):
            tempSentiment.insert(a, b)

    # print(tempSentiment)
    # print(_vectorOfSentiment)
    accuracy = accuracy_score(_vectorOfSentiment, tempSentiment)
    print(accuracy)

    # return accuracy

def KNN_Classification(_vectorOfOpinion, _vectorOfSentiment, _testSize, _nameOfOpinion, _K_Value):
    xTrain, xTest, yTrain, yTest = train_test_split(_vectorOfOpinion, _vectorOfSentiment, test_size=_testSize, random_state=1)
    neigh = KNeighborsClassifier(n_neighbors=_K_Value).fit(xTrain, yTrain)
    predicted = neigh.predict(xTest)
    accuracy = accuracy_score(yTest, predicted)*100
    
    # print("-"*70)
    print("Nilai akurasi %s dengan K:%d =" %(_nameOfOpinion, _K_Value), accuracy)
    # print("-"*70)

 

def main():
    startTime = time.time()
    randomStateList = []
    print("Reading TF-IDF...", end='', flush=True)
    # idSentiment, vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result/TF_IDF.csv') 
    idSentiment, vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result All Sentiment 5000/TF_IDF_2.csv')   
    print('done')

    SVM_Classification_With_CV(idSentiment, vectorOfOpinion, sentAsrama, 'Asrama', 1, 10)

    print("--- %s menit ---" % ((time.time() - startTime)/60))

main()