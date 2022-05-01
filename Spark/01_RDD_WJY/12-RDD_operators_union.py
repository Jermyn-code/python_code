# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd1 = sc.parallelize([1, 23, 4, 5, 6, 67, 7])
    rdd2 = sc.parallelize(["a", "b", "23"])

    # union 不进行去重，且可以进行不同的数据类型的来联合
    rdd3 = rdd1.union(rdd2)
    print(rdd3.collect())
