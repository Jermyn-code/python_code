# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/25 0025 22:51
# @Deception：
# @Filename : demo.py
# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([('a', 2), ('a', 8), ('a', 1), ('b', 1), ('b', 1)])

    rdd2 = rdd.groupByKey()
    print('带迭代器的形式：', rdd2.collect())


    # [('b', < pyspark.resultiterable.ResultIterable object at 0x7f1000857b50 >),
    #  ('a', < pyspark.resultiterable.ResultIterable object at 0x7f1000857ee0 >)
    #  ]

    def fun():
        for x in rdd2.collect():
            print(x[0])
            print(list(x[1]))
        return x[0], list(x[1])



    # print('解除嵌套的形式：', map(lambda x: (x[0], list(x[1])), rdd2.collect()))
    # [('b', [1, 1]), ('a', [2, 8, 1])]
