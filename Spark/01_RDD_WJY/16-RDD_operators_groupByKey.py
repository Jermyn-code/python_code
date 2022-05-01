# # coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize([('a', 2), ('a', 8), ('a', 1), ('b', 1), ('b', 1), ('a', 25)])
    rdd2 = rdd.groupByKey()
    print('带迭代器的形式：', rdd2.collect())


    # [('b', < pyspark.resultiterable.ResultIterable object at 0x7f1000857b50 >),
    #  ('a', < pyspark.resultiterable.ResultIterable object at 0x7f1000857ee0 >)
    # ]
    # def fun():
    #     for x in rdd2.collect():
    #         print(type(x))
    #         print(x[0])
    #         print(type(x[0]))
    #         print(x[1])
    #         print(type(x[1]))
    #         print("----------------------")
    #         for n in x[1]:
    #             print(n)
    #         print("----------------------")
    #         print(list(x[1]))
    #     return x[0], x[1]
    #
    #
    # fun()

    print('解除嵌套的形式：', rdd2.map(lambda x: (x[0], list(x[1]))).collect())
    # [('b', [1, 1]), ('a', [2, 8, 1])]

    # 通过 Values 中列表的元素的数量进行从大到小排序
    print(f"按列表的长度排序：{sorted(rdd2.mapValues(len).collect())}")

    # 通过 Values 中列表比较进行排序
    print(f"列表比较进行排序：{sorted(rdd2.mapValues(list).collect())}")
    # x[0] 值是 str 形式，x[1] 为一个可迭代类型
    # x[0] --> <class 'str'>
    # x[1] --> <pyspark.resultiterable.ResultIterable object at 0x7faf01246b20>
    # 将 list(x[1]) --> list 形式

    # rdd.mapValues()
