# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_map").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    rdd = sc.parallelize(["hadoop spark hello" "hadoop hadoop hello" "hello hello hello"])
    print(rdd.collect())

    print("----------------------------------")
    print(rdd.map(lambda line: line.split(" ")).collect())

    print(rdd.flatMap(lambda line: line.split(" ")).collect())
