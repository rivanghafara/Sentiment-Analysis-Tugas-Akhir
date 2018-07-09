import re, glob, nltk, string, csv
import numpy as np
import time

from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

def GetKataDasar(filename):
    with open(filename, 'r') as kataDasar:
        data = kataDasar.read().replace('\n', ',')
    return data

def TokenizingSplit(verse):
    return verse.split(' ')

def PunctuationRemoval(tokens):
    tokens = re.sub('[->)}:{",?+ !.(<;1234567890]','',str(tokens))
    tokens = re.sub('\n','',str(tokens))

    return tokens

# def GetSentimentOnly(dataset):
    

def Preprocessing(filename):
    semua = []
    words = []
    valueOfSentiment = []
    katadasar = GetKataDasar('katadasar.txt')
    with open(filename) as f:  # baca dataset csv 
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # print(row[0], len(row))
            if (row[0]):
                valueOfSentiment.append(int(row[-1]))
                tokenized = TokenizingSplit(row[0])
                temp = []
                for i in range(len(tokenized)):
                    punctuateRemoved = PunctuationRemoval(tokenized[i])
                    stopwordRemoved = stopword.remove(punctuateRemoved)
                    if (stopwordRemoved == ''):
                        continue
                    else:
                        stemmed = stemmer.stem(stopwordRemoved)
                        if (stemmed in katadasar):
                            temp.append(stemmed)
                            words.append(stemmed)
                        else:
                            continue
                semua.append(temp)
            else:
                continue
        # print(valueOfSentiment)
    return semua, words, valueOfSentiment

def GetExtraction(dataset):
    re_semua, __, valueOfSentiment = Preprocessing(dataset)

    listOfDokumen = {}
    for i in range(len(re_semua)):
        listOfDokumen[i] = Counter(re_semua[i])
    listOfString = []
    for _, val in listOfDokumen.items():  # melompati(loop over) _ di kamus
        for word, _ in val.items():
            if word not in listOfString:
                listOfString.append(word)  # append untuk menambah objek word baru kedalam list
    print("listOfString :",listOfString)
    print(len(listOfString))
        
    tempWord = [' ']
    tempWord.extend(listOfString)  # menambah string2 dari variable listOfString ke tempWord
    
    getListDF, valueTF = GetTF(re_semua, listOfString, tempWord)
    getListIDF = GetIDF(re_semua, getListDF, tempWord)
    TF_IDF(valueTF, getListIDF, tempWord, valueOfSentiment)

    # countDF = [] # Menghitung ada atau tidaknya kata pada setiap tanggapan

def GetTF(cleanDataset, listString, tempWord):
    countDF = []
    with open('Result/TF.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        reswriter.writerow(tempWord)  # write 1 row
        TF = []
        for i in range(len(cleanDataset)):
            rowjudul = "Tanggapan "+ str(i+1)
            #hitung pertanggapan
            words = cleanDataset[i]
            x = []
            tempCountDF = []
            count = 0
            for j in range(len(listString)):
                temp = listString[j]
                for k in range(len(words)):
                    if (temp == words[k]):
                        count += 1
                if (count > 0):
                    count = 1 + np.log(count)
                    tempCountDF.append(1)
                else:
                    tempCountDF.append(0)
                x.append(count)
                count = 0
            x.insert(0, rowjudul)
            TF.append(x)
            reswriter.writerow(x)
            countDF.append(tempCountDF)
        # print("temp DF : ", countDF)
    return countDF, TF


def GetIDF(cleanDataset, listDF, tempWord):
    with open('Result/IDF.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        reswriter.writerow(tempWord)  # write 1 row
        tempDF = []
        for i in range(len(listDF)):
            parent = []
            child = []
            subChild = listDF[i]
            if (i == 0):
                for j in range(len(subChild)):
                    tempDF.append(subChild[j])
            else:
                for j in range(len(subChild)):
                    parent.append(tempDF[j])
                    child.append(subChild[j])
                tempDF = []

            for k in range(len(parent)):
                tempDF.append(parent[k] + child[k])
        # print("DF : ", tempDF)

        IDF = []
        for j in range(len(tempDF)):
            if (tempDF[j] == 0):
                IDF.append(0)
            else:
                IDF.append(np.log(len(cleanDataset)/tempDF[j]))
        IDF.insert(0, "IDF")
        reswriter.writerow(IDF)
    return IDF


def TF_IDF(dataTF, dataIDF, tempWord, valueOfSentiment):
    with open('Result/TF_IDF.csv', 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        reswriter.writerow(tempWord)
        print("Value of Sentiment :" ,valueOfSentiment)
        # TF_IDF = []
        for k in range(len(dataTF)):
            rowjudul = "Tanggapan " + str(k+1)
            subChild = dataTF[k]
            sum_TFIDF = []
            temp = valueOfSentiment[k]
            for i in range(len(subChild)):
                if (i == 0):
                    sum_TFIDF.insert(0, rowjudul)
                else:
                    sum_TFIDF.append(subChild[i] * dataIDF[i])
                    # sum_TFIDF.insert(len(sum_TFIDF)+1, valueOfSentiment[k])
            # print(sum_TFIDF)
            sum_TFIDF.insert(len(subChild), temp)
            reswriter.writerow(sum_TFIDF)
            

    return sum_TFIDF

def main():
    startTime = time.time()
    # GetExtraction('Dataset-Training/Dataset-Training-5000.csv')
    GetExtraction('Dataset-Training/Dataset-Training-100-New.csv')
    print("--- %s seconds ---" % (time.time() - startTime))


main()