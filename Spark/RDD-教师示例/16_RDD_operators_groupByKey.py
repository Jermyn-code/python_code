# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([('a', 1), ('a', 1), ('a', 1), ('b', 1), ('b', 1)])

    rdd2 = rdd.groupByKey()

    print('带迭代器的形式：', rdd2.collect())
    print('解除嵌套的形式：', rdd2.map(lambda x: (x[0], list(x[1]))).collect())
