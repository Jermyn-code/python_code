import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib as plt
import numpy as np
from time import time
# 导入Tensorflow提供的读取MNIST的模块
import tensorflow.examples.tutorials.mnist.input_data as input_data

"""
1、数据准备
"""
# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
2、定义相同的模型结构
"""


# 定义全连接层函数
def fcn_layer(inputs,  # 输入数据
              input_dim,  # 上一层输入神经元数量
              output_dim,  # 当前层输出神经元数量
              activation=None):  # 激活函数
    # 以截断正态分布的随机数初始化W
    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))  # 以0初始化b
    XWb = tf.matmul(inputs, W) + b  # 建立矩阵叉乘表达式: inputs*W+b
    if activation is None:  # 默认不使用激活函数
        outputs = XWb
    else:  # 若传入激活函数，则用其对输出结果进行变换
        outputs = activation(XWb)
    return outputs


"(1)构建输入层"
x = tf.placeholder(tf.float32, [None, 784], name="X")

"(2)构建隐藏层1"
H1_NN = 256
h1 = fcn_layer(inputs=x,
               input_dim=784,
               output_dim=256,
               activation=tf.nn.relu)

"（3）构建输出层：计算输出结果"
forward = fcn_layer(inputs=h1,
                    input_dim=256,
                    output_dim=10,
                    activation=None)
pred = tf.nn.softmax(forward)

"""
3、训练模型
"""
"（1）定义标签数据占位符"
y = tf.placeholder(tf.float32, [None, 10], name="Y")
"（2）计算准确率"
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
4、还原模型
"""
"(1)设置模型文件的存放目录"
# 必须指定为模型文件的存放目录
ckpt_dir = "./ckpt_dir/"

"(2)读取还原模型"
# 创建saver
saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 在模型保存的目录下，从所有模型中找最新的模型
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:  # 模型中的数据存在，模型的路径存在
    # 从已保存的模型中读取参数，恢复到当前的session中去。
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restore model from" + ckpt.model_checkpoint_path)

"（3）模型评估"
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
