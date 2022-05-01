# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/29 0029 9:03
# @Deception：
# @Filename : 23-RDD_operators_first&take&count.py

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([1, 3, 5, 3, 1, 3, 2, 6, 7, 8, 6], 1)
    print(f"取出RDD的第一个元素:{rdd.first()}")
    print(f"取RDD的前N个元素，组合成list返回:{rdd.take(5)}")
    print(f"对RDD数据集进行降序排序，取前N个:{rdd.top(3)}")
    print(f"计算RDD有多少条数据，返回值是一个数字:{rdd.count()}")
