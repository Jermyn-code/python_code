import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
from time import time
# 导入Tensorflow提供的读取MNIST的模块
import tensorflow.examples.tutorials.mnist.input_data as input_data

"""
1、数据准备
"""
# 读取MNIST数据
mnist = input_data.read_data_sets("/", one_hot=True)

"""
2、构建模型
"""
"(1)定义标签占位符"
x = tf.placeholder(tf.float32, [None, 784], name="X")
y = tf.placeholder(tf.float32, [None, 10], name="Y")

"(2)构建隐藏层"

H1_NN = 256  # 第1隐藏层神经元为256个
H2_NN = 64  # 第2隐藏层神经元为64个
H3_NN = 32

# 输入层 到 第1隐藏层参数和偏置项
W1 = tf.Variable(tf.truncated_normal([784, H1_NN], stddev=0.1))
b1 = tf.Variable(tf.zeros([H1_NN]))

# 第1隐藏层 到 第2隐藏层参数和偏置项
W2 = tf.Variable(tf.truncated_normal([H1_NN, H2_NN], stddev=0.1))
b2 = tf.Variable(tf.zeros([H2_NN]))

# 第2隐藏层 到 第3隐藏层和偏置项
W3 = tf.Variable(tf.truncated_normal([H2_NN, H3_NN], stddev=0.1))
b3 = tf.Variable(tf.zeros([H3_NN]))

# 第3隐藏层 到 输出层参数和偏置项
W4 = tf.Variable(tf.truncated_normal([H3_NN, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

"（3）计算输出结果"
# 计算第1隐藏层结果
Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 计算第2隐藏层结果
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
# 计算第3隐藏层结果
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
# 计算输出结果
forward = tf.matmul(Y3, W4) + b4
pred = tf.nn.softmax(forward)

"""
3、训练模型
"""

"（1）定义交叉熵损失函数"
"改变前"
# loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

"改变后"
# TensorFlow提供了结合Softmax的交叉熵损失函数定义方法
# 用于避免因为log(0)值为NaN造成的数据不稳定
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

"（2）设置训练超参数"
train_epochs = 40
batch_size = 50
total_batch = int(mnist.train.num_examples / batch_size)
display_step = 1
learning_rate = 0.01

"（3）选择优化器"
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

"（4）计算准确率"
# 检查预测类别tf.argmax(pred,1) 与实际类别tf.argmax(y，1)的匹配情况
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"（5）迭代训练"
# 记录训练开始时间
startTime = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据,在读取前已经做好了洗牌工作
        sess.run(optimizer, feed_dict={x: xs, y: ys})  # 执行批次训练
    # total_batch个批次训练完成后，使用验证数据计算误差与准确率
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    if (epoch + 1) % display_step == 0:
        print("Train Epoch:", '%02d' % (epoch + 1),
              "Loss=", "{:.9f}".format(loss), "Accuracy", "{:.4}".format(acc))
# 显示运行总时间
duration = time() - startTime
print("Train Finished takes:", "{:.2f}".format(duration))

"（6）模型评估"
accu_test = sess.run(accuracy,
                     feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("Test Accuracy:", accu_test)

"""
4、模型应用和可视化
"""
"(1)模型预测"
# argmax能把pred值中最大值的下标给取出来，相当于转换为0~9数字
prediction_result = sess.run(tf.argmax(pred, 1),
                             feed_dict={x: mnist.test.images})
# 查看前10项预测结果
# print(prediction_result[0:10])

"（2）找出预测错误"
compare_lists = prediction_result == np.argmax(mnist.test.labels, 1)
# print(compare_lists)

"（3）找出预测错误样本的下标"
err_lists = [i for i in range(len(compare_lists)) if compare_lists[i] == False]
# print(err_lists)
# print(len(err_lists))

"（4）定义一个输出错误分类的函数"
import numpy as np


def print_predict_errs(labels,  # 标签值列表
                       prediction):  # 预测值列表
    count = 0
    compare_lists = (prediction == np.argmax(labels, 1))
    err_lists = [i for i in range(len(compare_lists)) if compare_lists[i] == False]
    for x in err_lists:
        print("index=" + str(x) +  # 错误样本的索引值
              "标签值=", np.argmax(labels[x]),
              "预测值=", prediction[x])
        count = count + 1
    print("总计：" + str(count))


print_predict_errs(labels=mnist.test.labels, prediction=prediction_result)

"（5）可视化查看预测错误的样本"


# 定义可视化函数
def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  prediction,  # 预测值列表
                                  index,  # 从第index个开始显示
                                  num=10):  # 默认一次显示10幅
    fig = plt.gcf()  # 获取当前图表，Get Current Figure
    fig.set_size_inches(10, 12)  # 设置图表大小，宽10英寸，高12英寸，1英寸等于2.54 cm
    if num > 25:
        num = 25  # 最多显示25个子图
    for i in range(0, num):  # 它针对每一幅图像它是怎么处理
        ax = plt.subplot(5, 5, i + 1)  # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')  # 显示第index个图像
        title = "label=" + str(np.argmax(labels[index]))  # 构建该图上要显示的title信息
        # if len(prediction) > 0:
        title += ",predict=" + str(prediction[index])  # title上面显示预测值
        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()


# 设置index出错能够显示在这，例如 610是错的，只能显示一个错的，并不能显示所有错误的，需要改进。
plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result, 610, 10)
