import csv
import math
import statistics
import matplotlib.pyplot as plt

# Loading Data dari CSV ke List
def loadData(namaFile):
    with open(namaFile) as csv_file:
        data = []
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append(
                    [
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                    ]
                )
            line_count += 1
        return data


# Scaling Data
def normalization(dataDiabetes):
    minMax = []
    for i in range(len(dataDiabetes[0]) - 1):
        kolom = []
        for row in dataDiabetes:
            kolom.append(row[i])
        minimum = min(kolom)
        maximum = max(kolom)
        minMax.append([minimum, maximum])

    for row in dataDiabetes:
        for i in range(len(row) - 1):
            row[i] = (float(row[i]) - float(minMax[i][0])) / (
                float(minMax[i][1]) - float(minMax[i][0])
            )
    return dataDiabetes


def standarization(dataDiabetes):
    minMax = []
    for i in range(len(dataDiabetes[0]) - 1):
        kolom = []
        for row in dataDiabetes:
            kolom.append(row[i])
        kolom = list(map(float, kolom))
        mean = statistics.mean(kolom)
        standardDeviation = statistics.stdev(kolom)
        minMax.append([mean, standardDeviation])

    for row in dataDiabetes:
        for i in range(len(row) - 1):
            row[i] = (float(row[i]) - float(minMax[i][0])) / (float(minMax[i][1]))
    return dataDiabetes


# 5 Fold Cross Validation
def foldClipping(listData):
    # data training set 1-614 dan testing set 615:768
    dataTrain1 = listData[0:613]
    dataTest1 = listData[614:767]

    # data training set 1-461 + 615-768 dan testing set 462-614
    dataTrain2 = listData[0:461] + listData[615:767]
    dataTest2 = listData[462:614]

    # data training set 1-307 + 462-768 dan testing set 308:461
    dataTrain3 = listData[0:306] + listData[461:767]
    dataTest3 = listData[307:460]

    # data training set 1-154 + 308-768 dan testing set 155:307
    dataTrain4 = listData[0:153] + listData[307:767]
    dataTest4 = listData[154:306]

    # data training set 155-768 dan testing set 1:154
    dataTrain5 = listData[154:767]
    dataTest5 = listData[0:153]
    return (
        dataTrain1,
        dataTrain2,
        dataTrain3,
        dataTrain4,
        dataTrain5,
        dataTest1,
        dataTest2,
        dataTest3,
        dataTest4,
        dataTest5,
    )


# Fungsi menghitung nilai prediksi ketetanggaan pilih salah-satu

# Euclidean
def hitungDistanceEuclidean(x, y):
    return math.sqrt(
        (float(x[0]) - float(y[0])) ** 2
        + (float(x[1]) - float(y[1])) ** 2
        + (float(x[2]) - float(y[2])) ** 2
        + (float(x[3]) - float(y[3])) ** 2
        + (float(x[4]) - float(y[4])) ** 2
        + (float(x[5]) - float(y[5])) ** 2
        + (float(x[6]) - float(y[6])) ** 2
        + (float(x[7]) - float(y[7])) ** 2
    )


# Manhattan
def hitungDistanceManhattan(x, y):
    return (
        abs((float(x[0]) - float(y[0])))
        + abs((float(x[1]) - float(y[1])))
        + abs((float(x[2]) - float(y[2])))
        + abs((float(x[3]) - float(y[3])))
        + abs((float(x[4]) - float(y[4])))
        + abs((float(x[5]) - float(y[5])))
        + abs((float(x[6]) - float(y[6])))
        + abs((float(x[7]) - float(y[7])))
    )


# Minkowski
def hitungDistanceMinkowski(x, y):
    return math.pow(
        (
            abs((float(x[0]) - float(y[0]))) ** (1 / 2)
            + abs((float(x[1]) - float(y[1]))) ** (1 / 2)
            + abs((float(x[2]) - float(y[2]))) ** (1 / 2)
            + abs((float(x[3]) - float(y[3]))) ** (1 / 2)
            + abs((float(x[4]) - float(y[4]))) ** (1 / 2)
            + abs((float(x[5]) - float(y[5]))) ** (1 / 2)
            + abs((float(x[6]) - float(y[6]))) ** (1 / 2)
            + abs((float(x[7]) - float(y[7]))) ** (1 / 2)
        ),
        1 / 2,
    )


# Fungsi Prediksi
def prediksi(nilaiK, dataTrain, dataTest):
    listPrediksi = []
    counter = 0
    for i in range(len(dataTest)):
        distance = []
        cekTetangga = [0, 0]  # Label untuk outcome 0,1
        for j in range(0, len(dataTrain)):
            # nilaiJarak dapat dirubah dengan perhitungan metode manhattan (hitungDistanceManhattan) dan minkowski (hitungDistanceMinkowski)
            nilaiJarak = hitungDistanceManhattan(dataTest[i], dataTrain[j])
            distance.append([nilaiJarak, dataTrain[j][8]])
        distance.sort()
        for d in range(0, nilaiK):
            if distance[d][1] == "0":
                cekTetangga[0] += 1
            elif distance[d][1] == "1":
                cekTetangga[1] += 1
        listPrediksi.append(cekTetangga.index(max(cekTetangga)))
    akurasi = accuracyDataset(listPrediksi, dataTest)
    return akurasi


# hitung akurasi dataset
def accuracyDataset(predictList, dataTest):
    count = 0
    for i in range(0, len(dataTest)):
        if str(predictList[i]) == dataTest[i][8]:
            count += 1
    return (count / len(dataTest)) * 100


# hitung akurasi rata-rata
def averageAccuracy(akurasi_1, akurasi_2, akurasi_3, akurasi_4, akurasi_5):
    return round(((akurasi_1 + akurasi_2 + akurasi_3 + akurasi_4 + akurasi_5) / 5), 5)


# hitung standar deviasi
def standarDeviasi(
    akurasi_1, akurasi_2, akurasi_3, akurasi_4, akurasi_5, akurasi_rataan
):
    return round(
        math.sqrt(
            ((akurasi_1 - akurasiRataan) ** 2 / 4)
            + ((akurasi_2 - akurasiRataan) ** 2 / 4)
            + ((akurasi_3 - akurasiRataan) ** 2 / 4)
            + ((akurasi_4 - akurasiRataan) ** 2 / 4)
            + ((akurasi_5 - akurasiRataan) ** 2 / 4)
        ),
        2,
    )


# inisialisasi list
dataDiabetes = []
dataTrain1 = []
dataTrain2 = []
dataTrain3 = []
dataTrain4 = []
dataTrain5 = []
dataTest1 = []
dataTest2 = []
dataTest3 = []
dataTest4 = []
dataTest5 = []
listAkurasi = []
listNilaiK = []
listKeseluruhanNilai = []
listStandarDeviasi = []


# loading data
dataDiabetes_awal = loadData("Diabetes.csv")
# untuk memilih jenis scaling dapat menggunakan normalization (normalisasi) atau standarization (standarisasi)
dataDiabetes = standarization(dataDiabetes_awal)

# membuat 5 dataset untuk 5 fold validation
(
    dataTrain1,
    dataTrain2,
    dataTrain3,
    dataTrain4,
    dataTrain5,
    dataTest1,
    dataTest2,
    dataTest3,
    dataTest4,
    dataTest5,
) = foldClipping(dataDiabetes)


for i in range(0, 41):
    # inisialisasi nilai k dengan iterasi
    k = i

    # inisialisasi akurasi nilaiPrediksi
    akurasiRataan = 0
    akurasi_1 = 0
    akurasi_2 = 0
    akurasi_3 = 0
    akurasi_4 = 0
    akurasi_5 = 0
    standardDeviasi = 0

    # hitung nilaiPrediksi dan Akurasi
    akurasi_1 = prediksi(k, dataTrain1, dataTest1)
    akurasi_2 = prediksi(k, dataTrain2, dataTest2)
    akurasi_3 = prediksi(k, dataTrain3, dataTest3)
    akurasi_4 = prediksi(k, dataTrain4, dataTest4)
    akurasi_5 = prediksi(k, dataTrain5, dataTest5)
    akurasiRataan = averageAccuracy(
        akurasi_1, akurasi_2, akurasi_3, akurasi_4, akurasi_5
    )
    standardDeviasi = standarDeviasi(
        akurasi_1, akurasi_2, akurasi_3, akurasi_4, akurasi_5, akurasiRataan
    )
    listKeseluruhanNilai.append([k, akurasiRataan, standardDeviasi])
    listStandarDeviasi.append(standardDeviasi)
    listAkurasi.append(akurasiRataan)
    listNilaiK.append(k)

posisiK = listAkurasi.index(max(listAkurasi))
print("Nilai dengan Akurasi Terbaik")
print("Nilai K : " + str(listNilaiK[posisiK]))
print("Nilai Akurasi : " + str(listAkurasi[posisiK]))
print("Nilai Standar Deviasi : " + str(listStandarDeviasi[posisiK]) + "\n")
posisiDeviasi = listStandarDeviasi.index(min(listStandarDeviasi))
print("Nilai Dengan Standar Deviasi Terendah")
print("Nilai K : " + str(listNilaiK[posisiDeviasi]))
print("Nilai Akurasi : " + str(listAkurasi[posisiDeviasi]))
print("Nilai Standar Deviasi : " + str(listStandarDeviasi[posisiDeviasi]))
print("\n")


# Membuat Plot Sebaran Nilai
plt.scatter(listNilaiK, listAkurasi)
plt.title("Scatter Plot Sebaran Akurasi dan K")
plt.xlabel("Nilai K")
plt.ylabel("Nilai Akurasi")
plt.show()

