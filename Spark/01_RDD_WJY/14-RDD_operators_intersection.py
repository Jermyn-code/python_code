# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_operator").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd1 = sc.parallelize([(100, "zhangsan"), (101, "lisi"), (102, "wangwu"),
                           (103, "zhaoliu")])
    rdd2 = sc.parallelize(
        [(100, "zhangsan"), (101, "lisi"), (102, "a"),
         (103, "lii"), (104, "wangwu"), (105, "zhaoliu")])
    rdd3 = rdd1.intersection(rdd2)
    print(rdd3.collect())
