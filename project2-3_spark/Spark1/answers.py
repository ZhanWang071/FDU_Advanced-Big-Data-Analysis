from pyspark import SparkContext, SparkConf
from operator import add, itemgetter
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data_path = './mernis/data_dump.sql'

# initialize spark
conf = SparkConf().setAppName("Project1").setMaster("local")
sc = SparkContext(conf=conf)

# load and clean the data
data = sc.textFile(data_path)
data = data.map(lambda x: x.split('\t'))
data = data.filter(lambda x: len(x) == 17)
print("The Number of Data: ", data.count())    # 49611709


################################################################################################
# E1. 统计土耳其所有公民中年龄最大的男性
################################################################################################
answer_E1 = data.filter(lambda x: x[6] == "E")\
            .sortBy(lambda x: int((x[8].split('/'))[2]))
print("\nAnswer for E1: ", answer_E1.first()[2]+' '+answer_E1.first()[3])


################################################################################################
# E2. 统计所有姓名中最常出现的字母
################################################################################################
data_E2 = data.map(lambda x: x[2] + x[3])
answer_E2 = data_E2.flatMap(lambda x: list(x)).map(lambda x: (x, 1)).reduceByKey(add).collect()
answer_E2 = [v for v in answer_E2 if v[0].isalpha()]
answer_E2.sort(key=lambda x:x[1], reverse=True)
print("\nAnswer for E2: ", answer_E2[0][0], " with frequency ", answer_E2[0][1])

# plt.clf()
# plt.bar(list(range(len(answer_E2))), [v[1] for v in answer_E2], tick_label=[str(v[0]) for v in answer_E2])
# plt.xlabel("Letter")
# plt.ylabel("Frequency")
# plt.savefig("./figs/E2.png")
# plt.show()


################################################################################################
# E3. 统计该国人口的年龄分布，年龄段分（0-18、19-28、29-38、39-48、49-59、>60）
################################################################################################
def group(x):
    age = int(2009-int((x[8].split("/"))[2]))
    if (age < 0):
        return ("others", 1)
    elif (age <= 18):
        return ("0-18", 1)
    elif (age <= 28):
        return ("19-28", 1)
    elif (age <= 38):
        return ("29-38", 1)
    elif (age <= 48):
        return ("39-48", 1)
    elif (age <= 59):
        return ("49-59", 1)
    else:
        return (">60", 1)
answer_E3 = data.map(group).reduceByKey(add).collect()
answer_E3.sort(key=lambda x:x[0][0])
print("\nAnswer for E3: ")
for (age, fre) in answer_E3:
    print(age, "\t",fre)

# plt.clf()
# plt.bar(list(range(len(answer_E3))), [v[1] for v in answer_E3], tick_label=[str(v[0]) for v in answer_E3])
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.savefig("./figs/E3.png")
# plt.show()


################################################################################################
# E4. 分别统计该国的男女人数，并计算男女比例
################################################################################################
male = data.filter(lambda x: x[6] == "E")
female = data.filter(lambda x: x[6] == "K")
male_counter = male.count()
female_counter = female.count()
print("\nAnswer for E4: ")
print(male_counter, " males and ", female_counter, " females")
print("Male:Female ", male_counter/female_counter)

################################################################################################
# E5. 统计该国男性出生率最高的月份和女性出生率最高的月份
################################################################################################
months = [str(i) for i in range(1, 13)]

answer_E5_E = male.map(lambda x: ((x[8].split("/"))[1], 1)).reduceByKey(add).collect()
answer_E5_E = [v for v in answer_E5_E if v[0] in months]
answer_E5_E.sort(key=lambda x:x[1], reverse=True)

answer_E5_K = female.map(lambda x: ((x[8].split("/"))[1], 1)).reduceByKey(add).collect()
answer_E5_K = [v for v in answer_E5_K if v[0] in months]
answer_E5_K.sort(key=lambda x:x[1], reverse=True)

print("\nAnswer for E5: ")
print("Male\t", answer_E5_E[0][0], " with the highest birth rate ", int(answer_E5_E[0][1]) / male_counter)
print("Female\t", answer_E5_K[0][0], " with the highest birth rate ", int(answer_E5_K[0][1]) / female_counter)

# answer_E5_E.sort(key=lambda x:int(x[0]))
# plt.clf()
# plt.bar(list(range(len(answer_E5_E))), [v[1] for v in answer_E5_E], tick_label=[v[0] for v in answer_E5_E])
# plt.title("Male")
# plt.xlabel("Months")
# plt.ylabel("Population")
# plt.savefig("./figs/E5_male.png")
# plt.show()

# answer_E5_K.sort(key=lambda x: int(x[0]))
# plt.clf()
# plt.bar(list(range(len(answer_E5_K))), [v[1] for v in answer_E5_K], tick_label=[v[0] for v in answer_E5_K])
# plt.title("Female")
# plt.xlabel("Months")
# plt.ylabel("Population")
# plt.savefig("./figs/E5_female.png")
# plt.show()


################################################################################################
# E6. 统计哪个街道居住人口最多
################################################################################################
answer_E6 = data.map(lambda x:(x[12], 1)).reduceByKey(add).collect()
answer_E6.sort(key=lambda x:x[1], reverse=True)
print("\nAnswer for E6: ", answer_E6[0][0])


################################################################################################
# N1. 分别统计男性和女性中最常见的10个姓
################################################################################################
answer_N1_E = male.map(lambda x: (x[3], 1)).reduceByKey(add).collect()
answer_N1_E.sort(key=lambda x:x[1], reverse=True)


answer_N1_K = female.map(lambda x: (x[3], 1)).reduceByKey(add).collect()
answer_N1_K.sort(key=lambda x:x[1], reverse=True)

print("\nAnswer for N1: ")
print("The most common last name of male: \n", [v[0] for v in answer_N1_E[:10]])
print("The most common last name of female: \n",[v[0] for v in answer_N1_K[:10]])


################################################################################################
# N2. 统计每个城市市民的平均年龄，统计分析每个城市的人口老龄化程度，判断当前城市是否处于老龄化社会
################################################################################################
city = data.map(lambda x: (x[9], 1)).reduceByKey(add).collect()
city_age = data.map(lambda x: (x[9], 2009-int((x[8].split("/"))[2]))).reduceByKey(add).collect()
answer_N2_avg = []
for i in city:
    for j in city_age:
        if (i[0] == j[0]):
            answer_N2_avg.append((i[0], j[1]/i[1]))
print("\nAnswer for N2: ")
print("Average age of cities: ")
for i in answer_N2_avg:
    print('%-15s %-f'%(i[0], i[1]))

city_60 = data.filter(lambda x: 2009-int((x[8].split("/"))[2]) > 60 )\
            .map(lambda x: (x[9], 1)).reduceByKey(add).collect()
city_65 = data.filter(lambda x: 2009-int((x[8].split("/"))[2]) > 65 )\
            .map(lambda x: (x[9], 1)).reduceByKey(add).collect()
answer_N2_old = []
for i in city_60:
    for j in city_65:
        if (i[0] == j[0]):
            answer_N2_old.append([i[0], i[1], j[1]])
for i in city:
    for j in answer_N2_old:
        if (i[0] == j[0]):
            j[1] /= i[1]
            j[2] /= i[1]
            if (j[1] > 0.1) | (j[2] > 0.07):
                j.append(True)
            else:
                j.append(False)
print("Aging cities or not: ")
for i in answer_N2_old:
    print('%-15s %-f %-f %-s'%(i[0], i[1], i[2], i[3]))


################################################################################################
# N3. 计算一下该国前10大人口城市中，每个城市的人口生日最集中分布的是哪2个月
################################################################################################
city_top10 = [v[0] for v in sorted(city, key=lambda x:x[1], reverse=True)[:10]]
answer_N3 = data.filter(lambda x: x[9] in city_top10)\
            .map(lambda x: ((x[9], (x[8].split("/"))[1]), 1)).reduceByKey(add).collect()

for i in range(len(answer_N3)):
    answer_N3[i] = [answer_N3[i][0][0], answer_N3[i][0][1], answer_N3[i][1]]
answer_N3.sort(key=itemgetter(0,2), reverse=True)

print("\nAnswer for N3: ")
for i in city_top10:
    tmp = 0
    for j in answer_N3:
        if (i == j[0]):
            print(i, j[1], j[2])
            tmp += 1
        if (tmp == 2):
            break


################################################################################################
# N4. 统计该国前10大人口城市中，每个城市的前3大姓氏，并分析姓氏与所在城市是否具有相关性
################################################################################################
answer_N4 = data.filter(lambda x: x[9] in city_top10)\
            .map(lambda x: ((x[9], x[3]), 1)).reduceByKey(add).collect()
for i in range(len(answer_N4)):
    answer_N4[i] = [answer_N4[i][0][0], answer_N4[i][0][1], answer_N4[i][1]]
answer_N4.sort(key=itemgetter(0,2), reverse=True)

print("\nAnswer for N4: ")
answer_N4_top3 = []
answer_N4.sort(key=itemgetter(0,2), reverse=True)
for i in city_top10:
    tmp = 0
    for j in answer_N4:
        if (i == j[0]):
            print('%-10s %-10s  %-d'%(i, j[1], j[2]))
            answer_N4_top3.append([i, j[1], j[2]])
            tmp += 1
        if (tmp == 3):
            break

city_top10_dict = {}
for i in city_top10:
    city_top10_dict[i] = city_top10.index(i)
last_name_dict = {}
tmp = 0
for j in answer_N4_top3:
    if j[1] not in last_name_dict.keys():
        last_name_dict[j[1]] = tmp
        tmp += 1

data_N4 = []
for v in answer_N4_top3:
    data_N4 += [(city_top10_dict[v[0]], Vectors.dense([last_name_dict[v[1]]])) for _ in range(v[2])]
spark =SparkSession(sc)
dataset = spark.createDataFrame(data_N4, ["label", "features"])
r = ChiSquareTest.test(dataset, 'features', 'label').head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))
