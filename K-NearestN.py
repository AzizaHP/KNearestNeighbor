import numpy
import math
import csv

#KAMUS DATA
dataTrain = []
dataTest = []
hasil = []
fold = 100
bagian = 4000//fold

#MEMBUKA FILE
def loadData():
    with open('DataTrain.txt') as fileF:
        dataset = numpy.genfromtxt('DataTrain.txt', dtype=None, delimiter=' ', names= True, skip_header=1)
    dataTrain = dataset[:4000]
    with open('DataTest.txt') as file:
        dataTesting = numpy.loadtxt(file, skiprows = 1, usecols=[x for x in range(1,5)])
    return dataTrain, dataTesting

#HITUNG EUCLIDEAN DISTANCE
def EuclideanDistance(x,y):
    hasil = 0.0
    for i in range(len(x)-1):
        hasil += (x[i] - y[i]) **2
    return round(math.sqrt(hasil),4)

#MENCARI TETANGGA DENGAN NILAI EUCLIDIAN TERENDAH DAN DIAMBIL SEBANYAK K TERATAS
def getNeighbor(train, test, k):
    jarak = []
    lNeighbor = []
    jjarak = []
    for x in train:
        temp = []
        temp.append(EuclideanDistance(x, test))
        temp.append(x)
        jarak.append(temp)
        jjarak.append(jarak)
    jjarak.sort()
    for x in range(k):
        lNeighbor.append(jjarak[x])
    return lNeighbor

#MENGKLASIFIKASIKAN DATA
def getClassification(neighbor):
    class1 = 0
    class0 = 0
    for x in range(len(neighbor)):
        if neighbor[x][1] == 1:
            class1+=1
        else:
            class0+=1
    if class1 >= class0:
        return 1
    else:
        return 0

#MELIHAT HASIL KLASIFIKASI
def getResult(trainSet, testSet, k):
    totalData = 0
    truee = 0
    classification = 0
    hoax = 0
    i = 1
    svData = []
    for data in testSet:
        neighbor = []
        neighbor = getNeighbor(trainSet, data, k)
        classification = getClassification(neighbor)
        svData.append([i, classification])
        i+=1
        totalData += 1
        if data.all() == classification:
            truee+=1
        akurasi = (truee/totalData)*100
        #print(data.any())
        print('TestingSet - %d, Nilai = %d, %d, %d, Klasifikasi = %d, Akurasi= %f, Hoax= %d' %
              (totalData, data[1], data[2], data[3], classification, akurasi, hoax))

#MAIN PROGRAM
dataTrain, dataTest = loadData()
getResult(dataTrain, dataTest, 75)