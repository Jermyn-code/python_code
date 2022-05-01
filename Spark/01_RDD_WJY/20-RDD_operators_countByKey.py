# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/29 0029 8:40
# @Deception：
# @Filename : 20-RDD_operators_countByKey.py
from pyspark import SparkConf, SparkContext
import json

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.textFile("../data/input/words.txt")
    rdd2 = rdd.flatMap(lambda x: x.split(" ")).map(lambda x: (x, 1))

    # 通过countByKey来对key进行计数, 这是一个Action算子
    result = rdd2.countByKey()

    print(result)
    print(type(result))
