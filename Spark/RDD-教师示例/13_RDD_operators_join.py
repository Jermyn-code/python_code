# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd1 = sc.parallelize([(1001, "zhangsan"), (1002, "lisi"), (1003, "wangwu"), (1004, "zhaoliu")])
    rdd2 = sc.parallelize([(1001, "销售部"), (1002, "研发部")])


    #1.通过join算来进行rdd之间的关联
    #2.对于join算子来说，关联条件按照二院元组中的key来进行关联的
    print(rdd1.join(rdd2).collect())

    #左外连接，右外链接，可以通过更换rdd的顺序类比于SQL中的表
    print(rdd1.leftOuterJoin(rdd2).collect())
