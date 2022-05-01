# % matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

"""
1、数据读取
"""
"（1）读取数据 "
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# mnist的shape
# print("Train image shape:",train_images.shape,"Train label shape:",train_labels.shape)
# print("Test image shape:",test_images.shape,"Test label shape :",test_labels.shape)

# 具体看一条数据
# print("image data:",train_images[1])

# 可视化
def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


# plot_image(train_images[20000])
# print(train_labels[20000])

"（2）划分数据集"
total_num = len(train_images)
valid_rates = 0.2  # 验证集占训练集的20%
train_num = int(total_num - total_num * valid_rates)

train_x = train_images[:train_num]  # 前部分给训练集
train_y = train_labels[:train_num]

valid_x = train_images[train_num:]  # 后部分给验证集
valid_y = train_labels[train_num:]

test_x = test_images
test_y = test_labels

print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)

"(3) 数据塑形"
# reshape的值有-1的话，会根据所给的新shape的信息，自动计算补足shape缺的值
# 把(28,28)的结构拉直为一行784。
train_x = train_x.reshape(-1, 784)
valid_x = valid_x.reshape(-1, 784)
test_x = test_x.reshape(-1, 784)
print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)

"(4)特征数据归一化"
train_x = tf.cast(train_x / 255.0, tf.float32)
valid_x = tf.cast(valid_x / 255.0, tf.float32)
test_x = tf.cast(test_x / 255.0, tf.float32)
# print(train_x[0])

"(5)对标签数据进行独热编码"
train_y = tf.one_hot(train_y, depth=10)
valid_y = tf.one_hot(valid_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)

# 独热编码调用方式
# x=[3,4]
# print(tf.one_hot(x,depth=10))
print(train_y)
"""
2、构建模型
"""
"(1) 构建模型"


def model(x, w, b):
    pred = tf.matmul(x, w) + b
    return tf.nn.softmax(pred)  # Softmax 会对每一类别估算出一个概率


"(2) 定义变量"
W = tf.Variable(tf.random.normal([784, 10], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros([10]), dtype=tf.float32)

"""
3、模型训练
"""
"(1)设置训练超参数"
training_epochs = 20
batch_size = 50  # 单次训练样本数(批次大小)
learning_rate = 0.001

"(2) 定义交叉熵损失函数"


def loss(x, y, w, b):
    pred = model(x, w, b)  # 计算模型预测值和标签值的差异
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)
    return tf.reduce_mean(loss_)  # 求均值，得出均方差


"(3)计算梯度"


# 计算样本数据[x, y]在参数[w, b]点上的梯度
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])  # 返回梯度向量


"(4)设置优化器"
# Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

"(5)计算准确率"


def accuracy(x, y, w, b):
    pred = model(x, w, b)  # 计算模型预测值和标签值的差异
    # 检查预测类别tf.argmax(pred,1) 与实际类别tf.argmax(y，1)的匹配情况
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 准确率，将布尔值转化为浮点数，并计算平均值
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"(6)迭代训练"
total_step = int(train_num / batch_size)  # 一轮训练有多少个批次，多少步
loss_list_train = []  # 用于保存训练集loss值的列表
loss_list_valid = []  # 用于保存验证集loss值的列表
acc_list_train = []  # 用于保存训练集Acc值的列表
acc_list_valid = []  # 用于保存验证集Acc值的列表

for epoch in range(training_epochs):
    for step in range(total_step):  # 一轮训练有多少个批次
        # 取出当前批次的数据xs,ys
        xs = train_x[step * batch_size:(step + 1) * batch_size]
        ys = train_y[step * batch_size:(step + 1) * batch_size]
        grads = grad(xs, ys, W, B)  # 根据当前批次的数据计算梯度
        # 根据优化器应用梯度，自动调整变量W, B
        optimizer.apply_gradients(zip(grads, [W, B]))
    # 求当前轮的损失（训练集、验证集）
    loss_train = loss(train_x, train_y, W, B).numpy()  # 计算当前轮训练损失
    loss_valid = loss(valid_x, valid_y, W, B).numpy()  # 计算当前轮验证损失
    acc_train = accuracy(train_x, train_y, W, B).numpy()
    acc_valid = accuracy(valid_x, valid_y, W, B).numpy()
    # 把当前轮得到的4个值添加到相应的列表中去。
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_valid)
    print("epoch={:3d},train_loss={:.4f},train_acc={:.4f},valid_loss={:.4f},val_acc={:.4f}".
          format(epoch + 1, loss_train, acc_train, loss_valid, acc_valid))
"(7)显示训练过程数据"
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.plot(loss_list_train,'blue',label="Train Loss")
# plt.plot(loss_list_valid,'red',label="Valid Loss")
# plt.legend(loc=1)#通过参数指定图例的位置
# plt.show()

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
# plt.plot(acc_list_train,'blue',label="Train Acc")
# plt.plot(acc_list_valid,'red',label= "Valid Acc")
# plt.legend(loc=1)#通过参数loc指定图例位置
# plt.show()

"(8)评估模型"
# 完成训练后，在测试集上评估模型的准确率
acc_test = accuracy(test_x, test_y, W, B).numpy()
print("Test accuracy:", acc_test)

"""
4、模型应用和可视化
"""
"(1)预测"


# 定义预测函数
def predict(x, w, b):
    pred = model(x, w, b)  # 计算模型预测值
    result = tf.argmax(pred, 1).numpy()
    return result


pred_test = predict(test_x, W, B)
print(pred_test[0])
"(2)可视化"
# 定义可视化函数
import matplotlib.pyplot as plt
import numpy as np


def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  preds,  # 预测值列表
                                  index,  # 从第index个开始显示
                                  num=10):  # 默认一次显示10幅
    fig = plt.gcf()  # 获取当前图表，Get Current Figure
    fig.set_size_inches(10, 4)  # 设置图表大小，宽10英寸，高4英寸，1英寸等于2.54 cm
    if num > 10:
        num = 10  # 最多显示10个子图
    for i in range(0, num):  # 它针对每一幅图像它是怎么处理
        ax = plt.subplot(2, 5, i + 1)  # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')  # 显示第index个图像
        title = "label=" + str(labels[index])  # 构建该图上要显示的title信息
        # if len(prediction) > 0:
        title += ",predict=" + str(preds[index])  # title上面显示预测值
        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()


plot_images_labels_prediction(test_images,
                              test_labels,
                              pred_test, 10, 10)
