# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test_map").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([1, 2, 3, 4, 5, 6], 3)

    def add(data):
        return data+10

    print(rdd.map(add).collect())


    #更简单的方式，就是定义lambda表达式来写一个匿名函数
    #print(rdd.map(lambda data: data + 10).collect())



