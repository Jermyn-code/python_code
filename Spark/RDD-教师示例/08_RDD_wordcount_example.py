# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    #1.读取文件，获取数据，创建RDD对象
    file_rdd = sc.textFile('../data/input/words.txt')

    #2.通过flatMap取出所有单词
    word_rdd = file_rdd.flatMap(lambda x: x.split(" "))

    #3.对已经取出的单词进行元组转换，key表示单词，value表示1
    word_with_one_rdd = word_rdd.map(lambda word: (word, 1))

    #4.使用reduceByKey对元组进行分组并聚合
    result_rdd = word_with_one_rdd.reduceByKey(lambda a, b: a + b)

    #5.通过collect()算子，将rdd的数据收集到Driver中，打印输出
    print(result_rdd.collect())

