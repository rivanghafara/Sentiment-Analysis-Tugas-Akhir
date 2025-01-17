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

def SVM_Classification_With_CV(_idSentiment, _vectorOfOpinion, _sentAsrama, _sentKesehatan, _sentAsuransi, _sentBeasiswa, _sentKegiatanM, _C, _C2, numFold):
       
    subset_size = int(len(_idSentiment)/numFold)
    tempSentiment = []
    tempSentimentKesehatan = []
    tempSentimentAsuransi = []
    tempSentimentBeasiswa = []
    tempSentimentKegiatanM = []

    for i in range(numFold):
        # print("================= Iterasi ",i,"=====================")
        idTest = _idSentiment[i * subset_size:][:subset_size]
        idTrain =  _idSentiment[:i * subset_size] + _idSentiment[(i+1)*subset_size:]
        
        xTrain = []
        xTest = []

        yTrainAsrama = []
        yTestAsrama = []

        yTrainKesehatan = []
        yTestKesehatan = []

        yTrainAsuransi = []
        yTestAsuransi = []

        yTrainBeasiswa = []
        yTestBeasiswa = []

        yTrainKegiatanM = []
        yTestKegiatanM = []

        #ambil data training
        for j in range(len(idTrain)):
            xTrain.append(_vectorOfOpinion[idTrain[j]])
            yTrainAsrama.append(_sentAsrama[idTrain[j]])
            yTrainKesehatan.append(_sentKesehatan[idTrain[j]])
            yTrainAsuransi.append(_sentAsuransi[idTrain[j]])
            yTrainBeasiswa.append(_sentBeasiswa[idTrain[j]])
            yTrainKegiatanM.append(_sentKegiatanM[idTrain[j]])


        #ambil data tsesting
        for k in range(len(idTest)):
            xTest.append(_vectorOfOpinion[idTest[k]])
            yTestAsrama.append(_sentAsrama[idTest[k]])
            yTestKesehatan.append(_sentKesehatan[idTest[k]])
            yTestAsuransi.append(_sentAsuransi[idTest[k]])
            yTestBeasiswa.append(_sentBeasiswa[idTest[k]])
            yTestKegiatanM.append(_sentKegiatanM[idTest[k]])

        clf = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrainAsrama)
        clfKesehatan = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrainKesehatan)
        clfAsuransi = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrainAsuransi)
        clfBeasiswa = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrainBeasiswa)
        clfKegiatanM = svm.SVC(kernel='linear', C=_C2).fit(xTrain, yTrainKegiatanM)

        predictedAsrama = clf.predict(xTest)
        predictedKesehatan = clfKesehatan.predict(xTest)
        predictedAsuransi = clfAsuransi.predict(xTest)
        predictedBeasiswa = clfBeasiswa.predict(xTest)
        predictedKegiatanM = clfKegiatanM.predict(xTest)
        
        
        for a, b, c, d, e, f in zip(idTest, predictedAsrama, predictedKesehatan, predictedAsuransi, predictedBeasiswa, predictedKegiatanM):
            tempSentiment.insert(a, b)
            tempSentimentKesehatan.insert(a, c)
            tempSentimentAsuransi.insert(a, d)
            tempSentimentBeasiswa.insert(a, e)
            tempSentimentKegiatanM.insert(a, f)

    # print(tempSentiment)
    # print(_sentAsrama)
    # accuracyAsrama = accuracy_score(_sentAsrama, tempSentiment)
    # accuracyKesehatan = accuracy_score(_sentKesehatan, tempSentimentKesehatan)
    # accuracyAsuransi = accuracy_score(_sentAsuransi, tempSentimentAsuransi)
    # accuracyBeasiswa = accuracy_score(_sentBeasiswa, tempSentimentBeasiswa)
    # accuracyKegiatanM = accuracy_score(_sentKegiatanM, tempSentimentKegiatanM)
    
    posAsrama, negAsrama = Percentage(tempSentiment)
    posKesehatan, negKesehatan = Percentage(tempSentimentKesehatan)
    posAsuransi, negAsuransi = Percentage(tempSentimentAsuransi)
    posBeasiswa, negBeasiswa = Percentage(tempSentimentBeasiswa)
    posKegiatanM, negKegiatanM = Percentage(tempSentimentKegiatanM)
    
    perAsrama = posAsrama, negAsrama
    perKesehatan = posKesehatan, negKesehatan
    perAsuransi = posAsuransi, negAsuransi
    perBeasiswa = posBeasiswa, negBeasiswa
    perKegiatanM = posKegiatanM, negKegiatanM



    # print(accuracyAsrama)
    # print(_C, accuracyAsrama, accuracyKesehatan, accuracyAsuransi, accuracyBeasiswa, accuracyKegiatanM)
    return perAsrama, perKesehatan, perAsuransi, perBeasiswa, perKegiatanM, len(_vectorOfOpinion)

def Percentage(_predictedSentiment):
    counter = 0
    for _, var in enumerate(_predictedSentiment):
        if (var == +1):
            counter += 1
        else:
            continue
        
    posPercentage = counter/len(_predictedSentiment)
    negPercentage = (len(_predictedSentiment)-counter)/len(_predictedSentiment)

    return posPercentage, negPercentage


def KNN_Classification(_idSentiment, _vectorOfOpinion, _vectorOfSentiment, _nameOfOpinion, numFold, _K_Value):
    subset_size = int(len(_idSentiment)/numFold)
    tempSentiment = []

    for i in range(numFold):
        # print("================= Iterasi ",i,"=====================")
        idTest = _idSentiment[i * subset_size:][:subset_size]
        idTrain =  _idSentiment[:i * subset_size] + _idSentiment[(i+1)*subset_size:]
        
        xTrain = []
        xTest = []

        yTrain = []
        yTestAsrama = []

        #ambil data training
        for j in range(len(idTrain)):
            xTrain.append(_vectorOfOpinion[idTrain[j]])
            yTrain.append(_vectorOfSentiment[idTrain[j]])

        #ambil data tsesting
        for j in range(len(idTest)):
            xTest.append(_vectorOfOpinion[idTest[j]])
            yTestAsrama.append(_vectorOfSentiment[idTest[j]])

        # clf = svm.SVC(kernel='linear', C=_C).fit(xTrain, yTrain)
        neigh = KNeighborsClassifier(n_neighbors=_K_Value).fit(xTrain, yTrain)

        predicted = neigh.predict(xTest)
        for a, b in zip(idTest, predicted):
            tempSentiment.insert(a, b)

    # print(tempSentiment)
    # print(_vectorOfSentiment)
    # accuracy = accuracy_score(_vectorOfSentiment, tempSentiment)
    posPercentage, negPercentage = Percentage(predicted)
    getPercentage = posPercentage, negPercentage
    # print(accuracy)

    return getPercentage
 
def WriteToCSV(_filename, _listing):
    header = 'Index', 'Asrama', 'Kesehatan', 'Asuransi', 'Beasiswa', 'Kegiatan Mahasiswa'
    with open(_filename, 'w', newline='') as file:
        reswrite = csv.writer(file, delimiter=',')
        reswrite.writerow(header)
        for _, var in enumerate(_listing):
            var = list(var)
            # var.insert(0, 'C = %s' %(i))
            reswrite.writerow(var)

def main():
    startTime = time.time()
    print("Reading TF-IDF...", end='', flush=True)
    # idSentiment, vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result/TF_IDF.csv') 
    idSentiment, vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM = ReadCSV('Result All Sentiment 5000/TF_IDF_2.csv')
    print('done')


    # a, b, c, d, e, lenData = SVM_Classification_With_CV(idSentiment, vectorOfOpinion, sentAsrama, sentKesehatan, sentAsuransi, sentBeasiswa, sentKegiatanM, 1, 2, 10)
    # print(a, b, c, d, e, lenData)
    result = KNN_Classification(idSentiment, vectorOfOpinion, sentKesehatan, 'Kesehatan', 10, 11)
    resultAsuransi = KNN_Classification(idSentiment, vectorOfOpinion, sentAsuransi, 'Asuransi', 10, 11)
    resultBeasiswa = KNN_Classification(idSentiment, vectorOfOpinion, sentBeasiswa, 'Beasiswa', 10, 11)
    resultKegiatanM = KNN_Classification(idSentiment, vectorOfOpinion, sentKegiatanM, 'KegiatanM', 10, 11)
    print(result)
    print(resultAsuransi)
    print(resultBeasiswa)
    print(resultKegiatanM)
    

    # getData = a, b, c, d, e
    # theResult.append(getData)

    # WriteToCSV('Hasil Pengujian/SVM.csv', theResult)
    print("--- %s menit ---" % ((time.time() - startTime)/60))

main()