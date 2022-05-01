# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([1, 2, 3, 4, 5, 6])

    #通过filter算子，过滤奇数
    rdd_filter = rdd.filter(lambda x: x % 2 == 1)

    print(rdd_filter.collect())
