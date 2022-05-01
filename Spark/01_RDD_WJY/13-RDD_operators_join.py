# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd1 = sc.parallelize([(100, "zhangsan"), (101, "lisi"), (102, "wangwu"), (103, "zhaoliu")])
    rdd2 = sc.parallelize([(101, "销售部"), (102, "研发部")])
    # 内连接
    print(rdd1.join(rdd2).collect())

    # 左连接
    print(rdd1.leftOuterJoin(rdd2).collect())
    # 右链接
    print(rdd1.rightOuterJoin(rdd2).collect())
