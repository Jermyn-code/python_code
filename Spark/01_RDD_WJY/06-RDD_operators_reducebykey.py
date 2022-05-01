# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_create").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    rdd = sc.parallelize([('a', 1), ('a', 1), ('b', 1), ('b', 1), ('b', 1)])

    print(rdd.reduceByKey(lambda a, b: a + b).collect())
    print("---------------------------------------------")
    print(rdd.reduce(lambda a, b: a + b))
