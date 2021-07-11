from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.sql import SQLContext, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression

data_path = './mernis/data_dump.sql'

################################################################################################
# N6. 计算前10大人口城市人口密度，其中城市的面积可Google搜索，面积单位使用平方千米
################################################################################################
def N6(data):
    print("\nN6")

    # top 10 largest cities
    city = data.map(lambda x: (x[9], 1)).reduceByKey(add).collect()
    city_top10 = [(v[0], v[1]) for v in sorted(city, key=lambda x:x[1], reverse=True)[:10]]

    # area of cities
    city_10_area = {'ISTANBUL':5343, 'KONYA':38873, 'IZMIR':11891, \
                    'ANKARA':24521, 'BURSA':1036,'SIVAS':2768, 'SAMSUN':1055, \
                    'AYDIN':1582, 'ADANA':1945, 'SANLIURFA':18584}
    
    # population density
    city_top10_pop_km2 = []
    for i in city_top10:
        city_top10_pop_km2.append((i[0], i[1]/city_10_area[i[0]]))

    print("The population density (number of people per square kilometer) of the top 10 largest cities: ")
    for i in city_top10_pop_km2:
        print('%-10s %-.4f'%(i[0],i[1]))


################################################################################################
# N7. 根据人口的出生地和居住地，分别统计土耳其跨行政区流动人口和跨城市流动人口占总人口的比例
################################################################################################
def N7(data):
    print("\nN7")
    regist_addr = data.map(lambda x: (x[9], x[10], x[11], x[12]))

    all_pop = data.count()
    migrant_city = regist_addr.filter(lambda x: x[0]!=x[2]).count()
    migrant_district = regist_addr.filter(lambda x: x[1]!=x[3]).count()

    print("inter-administrative migrant population / total population: %.4f"%(migrant_city/all_pop))
    print("inter-ctiy migrant population / total population: %.4f"%(migrant_district/all_pop))


################################################################################################
# H1. 某人所在城市的预测模型
################################################################################################
def H1(rawdata, sc):
    print("\nH1")

    sqlContext = SQLContext(sc)

    data = rawdata.map(lambda x: Row(address_city=x[11],address_district=x[12]))
    data_DF = sqlContext.createDataFrame(data)

    indexer = StringIndexer(inputCol="address_city", outputCol="label")
    indexed = indexer.fit(data_DF).transform(data_DF)
    indexer = StringIndexer(inputCol="address_district", outputCol="address_district_index")
    indexed = indexer.fit(indexed).transform(indexed)
    encoder = OneHotEncoder(inputCol = "address_district_index",outputCol="address_district_one")
    encoded = encoder.transform(indexed)
    assembler = VectorAssembler(inputCols=["address_district_one"],outputCol="features")
    data_h1 = assembler.transform(encoded)

    train, valid, test = data_h1.randomSplit([0.7, 0.1, 0.2], 2021)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)

    valid_predictions = model.transform(valid)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    valid_accuracy = evaluator.evaluate(valid_predictions)
    print("Valid set accuracy = %.4f" % (valid_accuracy))    # Valid set accuracy = 0.9780

    test_predictions = model.transform(test)
    data_probs = test_predictions.select("probability","label").rdd
    for i in range(1,6):
        print("Top %s accuracy = %.4f" % (i, topK_acc(data_probs, i)))
        # Top 1 accuracy = 0.9785                                                         
        # Top 2 accuracy = 0.9872                                                         
        # Top 3 accuracy = 0.9881                                                         
        # Top 4 accuracy = 0.9905                                                         
        # Top 5 accuracy = 0.9914 


def topK(probs,k):
    count = 0
    record = {}
    for i in probs:
        record[count] = i
        count = count + 1
    record = sorted(record.items(),key=lambda x:x[1],reverse=True)
    result = [record[i][0] for i in range(k)]
    return result

def topK_acc(data, k):
    topk_data = data.map(lambda x:(topK(x[0],k),x[1]))
    res = topk_data.filter(lambda x:int(x[1]) in x[0]).count() / float(topk_data.count())
    return res


################################################################################################
# H2. 性别预测模型
################################################################################################
def H2(rawdata, sc):
    print("\nH2")
    sqlContext = SQLContext(sc)

    data = rawdata.map(lambda x: Row(first_name=x[2],gender=x[6]))
    data_DF = sqlContext.createDataFrame(data)

    indexer = StringIndexer(inputCol="gender", outputCol="label")
    indexed = indexer.fit(data_DF).transform(data_DF)
    indexer = StringIndexer(inputCol="first_name", outputCol="first_name_index")
    indexed = indexer.fit(indexed).transform(indexed)
    encoder = OneHotEncoder(inputCol = "first_name_index",outputCol="first_name_one")
    encoded = encoder.transform(indexed)
    assembler = VectorAssembler(inputCols=["first_name_one"],outputCol="features")
    data_h2 = assembler.transform(encoded)

    train, valid, test = data_h2.randomSplit([0.7, 0.1, 0.2], 2021)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)

    valid_predictions = model.transform(valid)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    valid_accuracy = evaluator.evaluate(valid_predictions)
    print("Valid set accuracy = %.4f" % (valid_accuracy))    # Valid set accuracy = 0.9466  

    test_predictions = model.transform(test)
    test_accuracy = evaluator.evaluate(test_predictions)
    print("Test set accuracy = %.4f" % (test_accuracy))    # Test set accuracy = 0.9464 


################################################################################################
# H3. 姓名预测模型
################################################################################################
def H3(rawdata, sc):
    print("\nH3")
    sqlContext = SQLContext(sc)

    data = rawdata.map(lambda x: Row(last_name=x[3], city=x[9]))
    data_DF = sqlContext.createDataFrame(data)

    indexer = StringIndexer(inputCol="last_name", outputCol="label")
    indexed = indexer.fit(data_DF).transform(data_DF)
    indexer = StringIndexer(inputCol="city", outputCol="city_index")
    indexed = indexer.fit(indexed).transform(indexed)
    encoder = OneHotEncoder(inputCol = "city_index",outputCol="city_one")
    encoded = encoder.transform(indexed)
    assembler = VectorAssembler(inputCols=["city_one"],outputCol="features")
    data_h3 = assembler.transform(encoded)

    train, valid, test = data_h3.randomSplit([0.7, 0.1, 0.2], 2021)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)

    valid_predictions = model.transform(valid)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    valid_accuracy = evaluator.evaluate(valid_predictions)
    print("Valid set accuracy = %.4f" % (valid_accuracy))     # Valid set accuracy = 0.0139

    test_predictions = model.transform(test)
    data_probs = test_predictions.select("probability","label").rdd
    for i in range(1,6):
        print("Top %s accuracy = %.4f" % (i, topK_acc(data_probs, i)))


################################################################################################
# H4. 人口预测模型
################################################################################################
def H4(rawdata, sc):
    print("\nH4")
    sqlContext = SQLContext(sc)

    birth_pop = rawdata.map(lambda x: (int((x[8].split("/"))[2]), 1)).reduceByKey(add).collect()

    birth_pop_rdd = sc.parallelize([Row(features=Vectors.dense(i[0]), label=i[1]) for i in birth_pop])
    birth_pop_df = sqlContext.createDataFrame(birth_pop_rdd)

    train, valid, test = birth_pop_df.randomSplit([0.7, 0.1, 0.2], 2021)
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = lr.fit(train)
    predictions = model.transform(test)

    print(predictions.show())
    # +--------+-----+-------------------+
    # |features|label|         prediction|
    # +--------+-----+-------------------+
    # |[1910.0]|    4|-114.49577254327596|
    # |[1919.0]|   29| 44.130558155986364|
    # |[1922.0]|   39|  97.00600172240229|
    # |[1932.0]|  265| 273.25748027713416|
    # |[1937.0]|  303| 361.38321955450374|
    # |[1939.0]|  316|  396.6335152654501|
    # |[1940.0]|  362| 414.25866312091966|
    # |[1952.0]|  560|  625.7604373865979|
    # |[1953.0]|  626|  643.3855852420747|
    # |[1960.0]|  960|  766.7616202303834|
    # |[1971.0]| 1056|  960.6382466405921|
    # |[1980.0]| 1358| 1119.2645773398472|
    # |[1984.0]| 1259|   1189.76516876174|
    # |[1988.0]| 1050| 1260.2657601836327|
    # +--------+-----+-------------------+


if __name__=="__main__":

    # initialize spark
    conf = SparkConf().setAppName("Spark2").setMaster("local[*]")    # use all threads
    sc = SparkContext(conf=conf)

    # load and clean the data
    data = sc.textFile(data_path)
    data = data.map(lambda x: x.split('\t'))
    data = data.filter(lambda x: len(x) == 17)
    # print("The Number of Data: ", data.count())    # 49611709
    data = data.filter(lambda x: int(2009 - int((x[8].split('/'))[2])) <= 150)
    # print("The Number of Data: ", data.count())    # 49611216

    data_1, data_99 = data.randomSplit([0.001,0.999], 2021)
    # print("The Number of Data: ", data_1.count())    # 49351

    N6(data)
    N7(data)
    H1(data_1, sc)
    H2(data_1, sc)
    H3(data_1, sc)
    H4(data_1, sc)
