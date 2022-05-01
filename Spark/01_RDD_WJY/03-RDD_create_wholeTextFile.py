# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_create").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    tiny_file_rdd = sc.wholeTextFiles("../data/input/tiny_files")

    print(tiny_file_rdd.map(lambda x: x[2].collect()))

