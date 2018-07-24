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

    return sentimentList, valueOfSentimentAsrama, valueOfSentimentKesehatan, valueOfSentimentAsuransi, valueOfSentimentBeasiswa, valueOfSentimentKegiatanMahasiswa

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
    # print("-"*70)
    print("Nilai akurasi %s =" %(_nameOfOpinion), accuracy)
    # print("-"*70)
    # print("Nilai akurasi %s dengan C:%d =" %(_nameOfOpinion, _C), accuracy)

    return accuracy

def KNN_Classification(_vectorOfOpinion, _vectorOfSentiment, _testSize, _nameOfOpinion, _K_Value):
    xTrain, xTest, yTrain, yTest = train_test_split(_vectorOfOpinion, _vectorOfSentiment, test_size=_testSize, random_state=1)
    neigh = KNeighborsClassifier(n_neighbors=_K_Value).fit(xTrain, yTrain)
    predicted = neigh.predict(xTest)
    accuracy = accuracy_score(yTest, predicted)*100
    
    # print("-"*70)
    print("Nilai akurasi %s dengan K:%d =" %(_nameOfOpinion, _K_Value), accuracy)
    # print("-"*70)

def SVM_Classification_With_CV(_vectorOfOpinion, _vectorOfSentiment, _nameOfOpinion, _C):
    rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    clf = svm.SVC(kernel='linear', C=_C)
    result = cross_val_score(clf, _vectorOfOpinion, _vectorOfSentiment, cv=rs)
    print('Akurasi SVM %s dengan CV:10 dan C:%d =' %(_nameOfOpinion, _C), result, 'Max Result =', max(result))


def main():
    startTime = time.time()
    randomStateList = []
    print("Reading TF-IDF...", end='', flush=True)
    # vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result/TF_IDF.csv') 
    vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result All Sentiment 5000/TF_IDF.csv')   
    print('done')

    # SVM_Classification_With_CV(vectorOfOpinion, sentAsrama, 'Asrama', 1)
    # SVM_Classification(vectorOfOpinion, sentAsrama, 0.1, 'Asrama', 1)
    
    SVM_Classification(vectorOfOpinion, sentAsrama, 0.1, 'Asrama', 1)
    
    # for i in range(1, 11):

    #     print('Nilai C = %d....' %(i), end='', flush=True)
    #     a = SVM_Classification(vectorOfOpinion, sentAsrama, 0.1, 'Asrama', i, 12)
    #     b = SVM_Classification(vectorOfOpinion, sentKesehatan, 0.1, 'Kesehatan', i, 12)
    #     c = SVM_Classification(vectorOfOpinion, sentAsuransi, 0.1, 'Asuransi', i, 12)
    #     d = SVM_Classification(vectorOfOpinion, sentBeasiswa, 0.1, 'Beasiswa', i, 12)
    #     e = SVM_Classification(vectorOfOpinion, sentKegiatanM, 0.1, 'KegiatanM', i, 12)
    #     seq = a, b, c, d, e
    #     randomStateList.append(seq)
    #     print('done')
    
    # for j in range(20, 110, 10):
    #     print('Nilai C = %d....' %(j), end='', flush=True)
    #     a = SVM_Classification(vectorOfOpinion, sentAsrama, 0.1, 'Asrama', j, 12)
    #     b = SVM_Classification(vectorOfOpinion, sentKesehatan, 0.1, 'Kesehatan', j, 12)
    #     c = SVM_Classification(vectorOfOpinion, sentAsuransi, 0.1, 'Asuransi', j, 12)
    #     d = SVM_Classification(vectorOfOpinion, sentBeasiswa, 0.1, 'Beasiswa', j, 12)
    #     e = SVM_Classification(vectorOfOpinion, sentKegiatanM, 0.1, 'KegiatanM', j, 12)
    #     seq = a, b, c, d, e
    #     randomStateList.append(seq)
    #     print('done')

    # WriteToCSV('Perbandingan Random State.csv', randomStateList)

    print("--- %s menit ---" % ((time.time() - startTime)/60))

main()