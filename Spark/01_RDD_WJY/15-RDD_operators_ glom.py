# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_operator").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    """
    功能：将RDD的数据，加上嵌套，这个嵌套按照分区来进行比如RDD数据
    [1,2,3,4,5]有2个分区那么，被glom后，数据变成：[ 	[1,2,3], 	[4,5]	]
    """
    rdd1 = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8])
    print(rdd1)
    print(rdd1.collect())

    # 加嵌套
    """
        >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
        >>> sorted(rdd.glom().collect())
        [[1, 2], [3, 4]]
    """
    print(rdd1.glom().collect())

    # 解嵌套
    print(rdd1.glom().flatMap(lambda x: x).collect())
