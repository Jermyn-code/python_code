# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([1, 2, 3, 4, 5, 2, 3, 4, 2, 3, 4, ])
    distinct_rdd = rdd.distinct()
    print(distinct_rdd.collect())

    rdd2 = sc.parallelize([('a', 1), ('a', 1), ('b', 1)])
    distinct_rdd2 = rdd2.distinct()
    print(distinct_rdd2.collect())
