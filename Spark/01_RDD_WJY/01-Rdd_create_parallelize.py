# coding:utf-8

from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
    # 初始化执行环境，创建 Sparkcontext 对象
    conf = SparkConf().setAppName("test_create").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    # 演示通过并行化创建RDD,本地集合-->分布式对象集合转化
    rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # rdd1 = sc.parallelize([1, 2, 3], 3)

    # 查看默认分区
    print("默认分区：", rdd.getNumPartitions())

    # print("指定分区：", rdd1.getNumPartitions())
    # collect() 方法是将RDD中的每个分区的数据，都发送到Driver中，形成一个Python list对象输出
    print(rdd.collect())
