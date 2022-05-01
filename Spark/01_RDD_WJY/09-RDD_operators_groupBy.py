# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # 通过 groupBy 对数据进行分组，传入函数的意思是:通过这个函数，确定按照谁来进行分组（返回谁就可以）
    # 分组规则和 SQL 是一致的
    rdd = sc.parallelize([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)])

    group_rdd = rdd.groupBy(lambda x: x[0])
    print(group_rdd.collect())
    print(group_rdd.map(lambda x: (x[0], list(x[1]))).collect())
