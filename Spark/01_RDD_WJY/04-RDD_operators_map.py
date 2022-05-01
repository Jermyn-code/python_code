# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_map").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9])


    def fun(rdd):
        return rdd + 10


    print(rdd.map(fun).collect())
    print(rdd.map(lambda x: x + 10).collect())
