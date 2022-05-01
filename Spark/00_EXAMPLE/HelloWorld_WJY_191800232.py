# coding:utf-8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 1.创建SparkConf对象
    conf = SparkConf().setAppName("HelloWorld_WJY_191800232").setMaster("local[*]")

    # 2.通过SparkConf对象构建SparkContext对象
    sc = SparkContext(conf=conf)

    # 需求:wordcount单词计数，读取HDFS上的words.txt文件，对这个文件内部的单词出现次数进行统计，最终得到每个单词的出现数目

    # 3.读取文件
    file_rdd = sc.textFile("hdfs://master:8020/input/words.txt")

    # 4.将单词进行分割，得到一个存储全部单词的集合对象
    words_rdd = file_rdd.flatMap(lambda line: line.split(" "))

    # 5.把单词给转换为元组对象，key是单词，value是数字1
    words_with_one_rdd = words_rdd.map(lambda x: (x, 1))

    # 6.通过将元组的value按照key来进行分组，对所有的value进行聚合操作(加法操作)
    result_rdd = words_with_one_rdd.reduceByKey(lambda a, b: a + b)

    # 7.通过collect方法收集RDD的数据进行结果输出
    print(result_rdd.collect())
