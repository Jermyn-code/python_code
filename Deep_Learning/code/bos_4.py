import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.utils import shuffle #洗牌，把样本打乱
from sklearn.preprocessing import scale
import matplotlib. pyplot as plt
'进行归一化处理'
'''
1、数据准备
'''
#读取数据文件
#pd.set_option('display.max_columns',None)打印出完整的信息
df = pd.read_csv('boston.csv',header=0)
#显示数据摘要描述信息
#print(df.describe())

#获取df的值
df=df.values
#为了后面方便处理，转为np数组格式
df=np.array(df)
print(df.shape)

#对特征数据[0到11]列做(0-1)归一化，标签结果不用归一化
for i in range(12):
    df[:,i]=df[:, i]/(df[:,i].max()-df[:,i].min())

# x_data为归一化后的前12列特征数据,y_data为最后1列标签数据
#np.set_printoptions(threshold=np.inf)打印出完整的信息
#把样本的特征数据跟标签数据完整读进来
x_data = df[:,:12]
print(x_data.shape)
y_data = df[:,12]
print(y_data.shape)
'''
2、构建模型
'''
# 定义特征数据和标签数据的占位符
x =tf.placeholder(tf. float32,[None, 12],name="X")# 12个特征数据(12列)
y =tf.placeholder(tf. float32,[None, 1],name="Y")# 1个标签数据(1列)

#定义了一个命名空间，相当于给下边语句打个包
with tf.name_scope("Model"):
    # w初始化值为shape=(12,1)的正态分布数，标准差为0.01
    w =tf. Variable(tf.random_normal([12, 1],stddev=0.01),name="W")
    # b初始化值为1.0
    b = tf. Variable(1.0,name="b")
    # w和x是矩阵叉乘，用matmul,不能用mutiply或者*
    def model(x,w,b):#使用多元线性方程构建模型，这个模型的形态是矩阵相乘
        return tf. matmul(x, w) + b
    #预测计算操作，前向计算节点
    pred=model(x,w,b)
'''
3、训练模型
'''
#设置训练超参数
train_epochs = 50#迭代轮次
learning_rate = 0.01#学习率
#定义均方差损失函数
with tf.name_scope ("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred, 2))
#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#声明会话
sess = tf. Session()
#定义初始化变量的操作
init = tf. global_variables_initializer()
#启动会话
sess.run(init)

#设置日志存储目录
logdir='./log_bos'
#创建一个操作，用于记录损失值loss, 后面在TensorBoard中SCALARS栏可见
sum_loss_op = tf. summary. scalar("loss",loss_function)
#把所有需要记录摘要日志文件的合并，方便一次性写入
merged = tf. summary. merge_all()
#创建摘要的文件写入器(FileWriter)
#创建摘要writer，将计算图写入摘要文件，后面在TensorBoard中GRAPHS栏可见
writer = tf.summary.FileWriter(logdir,sess.graph)
'''
迭代训练
'''
for epoch in range(train_epochs):
    loss_sum= 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1, 1)
        _,summary_str, loss = sess.run([optimizer, sum_loss_op, loss_function], feed_dict={x: xs, y: ys})
        writer. add_summary(summary_str, epoch)
        loss_sum = loss_sum + loss
        #打乱数据顺序
    x_data,y_data = shuffle(x_data,y_data)
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    print(" epoch=", epoch+1, "loss=", loss_average, "b=", b0temp, "w=", w0temp)
#关闭session
sess.close()

#tensorboard --logdir=/Users/lvshuang/Downloads/Learning_TensorFlow/log_bos
