""""""
import matplotlib. pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
"""
1、MNIST手写数字识别数据集"
"""
"（1）加载数据集"
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"（2）了解MNIST手写数字识别数据集"
print("训练集train数量:", mnist.train. num_examples,
      "验证validation数量：",mnist.validation.num_examples,
       "测试集test数量:", mnist.test.num_examples)
print("train images shape:",mnist.train.images.shape,
        "labels shape:",mnist.train.labels.shape)

"（3）具体看一幅image的数据"
print(len(mnist.train.images[0]))#5500张训练数据中的第一幅图像。
print(mnist.train.images[0].shape)#一维数组
print(mnist.train.images[0])
"（4）可视化"
#显示图像 按照行和列的模式，把图像的数据做了一个重新的整理。
print(mnist.train.images[0].reshape(28, 28))
#可视化
# def plot_image(image):
#     plt.imshow(image.reshape(28, 28),cmap='binary')
#     plt.show()
# plot_image(mnist.train.images[0])
# plot_image(mnist.train.images[1])
# plot_image(mnist.train.images[20000])
"""
2、学习reshape
"""
import numpy as np
int_array =np.array([i for i in range(64)])
# print(int_array)
# print(int_array.reshape(8,8))
# print(int_array.reshape(4,16))

#plt.imshow(mnist.train.images[20000].reshape(14,56),cmap="binary")
#plt.show()
"""
3、认识标签 独热编码
"""
"（1）显示独热编码"
# print(mnist.train.labels[1])  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
"（2）取出独热编码"
#print(np.argmax(mnist.train.labels[1]))
"（3）不使用独热编码"
#不使用独热编码,训练集中前5个样本的标签值
mnist_no_one_hot = input_data.read_data_sets("MNIST_data/", one_hot=False)
# print(mnist_no_one_hot.train.labels[0:5])#[7 3 4 6 1]
# print(plot_image(mnist.train.images[0]))
"""
4、读取（训练、验证、测试）数据
"""
# print("train images shape:",mnist.train.images.shape,
#         "labels shape:",mnist.train.labels.shape)
# print("validation images shape:",mnist.validation.images.shape,
#         "labels shape:",mnist.validation.labels.shape)
# print("test images shape:",mnist.test.images.shape,
#         "labels shape:",mnist.test.labels.shape)
"""
5、数据的批量读取
"""
#切片
#print(mnist.train.labels[0:10])

#批量读取样本的方法 next_batch
batch_images_xs,batch_labels_ys=mnist.train.next_batch(batch_size=10)
print(batch_labels_ys)
#print(batch_images_xs.shape,batch_labels_ys.shape)

#再执行一遍的话，会接着上一次往后去取10条，再取的话是会重新洗牌的
batch_images_xs1,batch_labels_ys1=mnist.train.next_batch(batch_size=10)
print(batch_labels_ys1)

"""
6、模型构建
"""
"1、定义待输入数据的占位符"
# mnist中每张图片共有28*28=784个像素点
x =tf.placeholder(tf. float32,[None,784],name="X")
# 0-9一共10个数字=> 10个类别
y =tf.placeholder(tf. float32,[None,10],name="Y")

"2、定义模型变量"
W = tf. Variable(tf.random_normal([784, 10]), name="W")#符合正态分布的随机数
b = tf. Variable(tf.zeros([10]),name="b")

#了解一下正态分布
norm = tf. random_normal([100])
# with tf.Session() as sess:
#     norm_data = norm.eval()
#     print(norm_data[:10])#打印前10个随机数
#以直方图模式打印数据
# plt.hist(norm_data)
# plt.show()

"3、定义前向运算"
forward=tf.matmul(x, W) +b
"4、结果分类"
pred = tf.nn.softmax(forward)

# 了解一下softmax
x = np.array([[-3.1,1.8,9.7,-2.5]])
pred1 = tf.nn.softmax(x) # Softmax分类
sess=tf.Session()
v=sess.run(pred1)
print(v)
sess.close()
"5、定义损失函数"
loss_function=tf.reduce_mean(- tf. reduce_sum(y*tf.log(pred),reduction_indices=1))
