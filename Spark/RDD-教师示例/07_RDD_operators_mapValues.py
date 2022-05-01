# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([('a',1), ('b',2), ('c',3), ('d',4), ('e',5)])

    rdd1 = rdd.map(lambda x: x * 10)

    print('rdd1:',rdd1.collect())

    rdd3 = rdd.map(lambda x:(x[0],x[1]*10))
    print('rdd3:',rdd3.collect())


    rdd2 = rdd.mapValues(lambda x: x * 10)
    print('rdd2:',rdd2.collect())



