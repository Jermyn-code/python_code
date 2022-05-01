# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/26 0026 14:04
# @Deception：需求:读取data文件夹中的order.text文件，提取北京的数据组合北京和商品类别进行输出同时对结果集进行去重，得到北京售卖的商品类别信息
# @Filename : 19-RDD_example(modify-01).py


from pyspark import SparkConf, SparkContext
import json
import os

os.environ['HADOOP_CONF_DIR'] = "/export/server/hadoop/etc/hadoop"

if __name__ == '__main__':
    # conf = SparkConf().setAppName("test").setMaster("local[*]")
    conf = SparkConf().setAppName("test-order0yarn").setMaster("yarn")
    sc = SparkContext(conf=conf)

    # file_rdd = sc.textFile("../data/input/order.text")
    file_rdd = sc.textFile("hdfs://master:8020/input/order.text")
    rdd1 = file_rdd.flatMap(lambda line: line.split("|"))
    print("-" * 200)
    print(f"rdd1:对数据进行扁平化处理，去除 '|',得到 json 类型字符串\n{rdd1.collect()}")
    rdd2 = rdd1.map(lambda x: json.loads(x))
    print("-" * 200)
    print(f"rdd2:json字符串到字典对象的转换\n{rdd2.collect()}")
    rdd3 = rdd2.filter(lambda x: x["areaName"] == "北京")
    print("-" * 200)
    print(f"rdd3:使用filter算子过滤出'areaName' == '北京'的数据\n{rdd3.collect()}")
    rdd4 = rdd3.map(lambda x: x['areaName'] + "_" + x['category'])
    print("-" * 200)
    print(f"rdd4:组合北京 和 商品类型形成新的字符串\n{rdd4.collect()}")
    rdd5 = rdd4.distinct()
    print("-" * 200)
    print(f"rdd5:去重\n{rdd5.collect()}")
