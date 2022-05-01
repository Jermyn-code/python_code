# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([('a',1),('a',1),('b',1),('b',2),('b',3)])

    #通过groupBy对数据进行分组，传入函数的意思是：通过这个函数，确定按照谁来进行分组（返回谁就可以）
    #分组规则和SQL是一致的，HASH

    group_rdd = rdd.groupBy(lambda x: x[0])

    print(group_rdd.collect())

    map_rdd = group_rdd.map(lambda x: (x[0], list(x[1])))
    print(map_rdd.collect())
