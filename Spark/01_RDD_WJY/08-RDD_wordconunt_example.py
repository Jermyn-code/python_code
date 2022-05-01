# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test_create").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # 读取数据，获取数据，创建RDD对象
    # file_rdd = sc.textFile('D:\\学习\\2022学期\\python_code\\Spark\\data\\input\\words.txt')
    file_rdd = sc.textFile('../data/input/words.txt')

    print("-" * 100)
    print(f"导出文件内容为\n{file_rdd.collect()}")
    # 通过 flatMap 取出单词
    print("-" * 100)
    word_rdd = file_rdd.flatMap(lambda x: x.split(" "))
    print(f"扁平化处理后去除 空格\n{word_rdd.collect()}")
    # 对已经取出的单词进行元组转换，key 表示单词，value 表示 1
    word_with_one_rdd = word_rdd.map(lambda word: (word, 1))
    print("-" * 100)
    print(f"map操作形成 （'word'，1）\n{word_with_one_rdd.collect()}")
    # 使用 reduceByKey 对元组进行分组聚合
    result_rdd = word_with_one_rdd.reduceByKey(lambda a, b: a + b)

    print("-"*100)
    print(f"reduceByKey操作\n{result_rdd.collect()}")
