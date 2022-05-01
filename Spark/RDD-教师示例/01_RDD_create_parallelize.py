#coding:utf8

#导入spark的相关包
from pyspark import SparkConf,SparkContext

if __name__ == '__main__':

    # 初始化执行环境，构建SparkContext对象
    conf = SparkConf().setAppName("test_create1").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    #演示通过并行化集合的方式去创建RDD，本地集合->分布式对象的转化
    rdd1 = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 查看默认分区
    print("默认分区数：",rdd1.getNumPartitions())

    rdd2 = sc.parallelize([1, 2, 3],3)
    print("分区数：",rdd2.getNumPartitions())

    # collect方法，是将RDD中每个分区的数据，都发送到Driver中，形成一个Python LIst对象进行输出的
    print(rdd1.collect())






