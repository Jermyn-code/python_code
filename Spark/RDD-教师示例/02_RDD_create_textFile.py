# coding:utf8

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    # 构建SparkContext对象
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    #通过textFile API读取数据

    #读取本地数据文件
    file_rdd1 = sc.textFile("../data/input/words.txt")
    print("默认分区数",file_rdd1.getNumPartitions())
    print("file_rdd1内容", file_rdd1.collect())

    #加入最小分区参数的测试
    file_rdd2= sc.textFile("../data/input/words.txt",3)
    file_rdd3 = sc.textFile("../data/input/words.txt", 100)
    print("file_rdd2 分区数:", file_rdd2.getNumPartitions())
    print("file_rdd3 分区数:", file_rdd3.getNumPartitions())

    # 读取HDFS文件数据测试
    hdfs_rdd = sc.textFile("hdfs://node1:8020/input/words.txt")
    print("hdfs_rdd 内容:", hdfs_rdd.collect())




