import matplotlib. pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
"""
1、载入数据
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
2、构建模型
"""
"（1）定义待输入数据x和y的占位符"
# mnist中每张图片共有28*28=784个像素点
x =tf.placeholder(tf. float32,[None,784],name="X")
# 0-9一共10个数字=> 10个类别
y =tf.placeholder(tf. float32,[None,10],name="Y")
"（2）定义模型变量"
W = tf. Variable(tf.random_normal([784, 10]), name="W")#符合正态分布的随机数
b = tf. Variable(tf.zeros([10]),name="b")
"（3）用单个神经元构建神经网络"
forward=tf.matmul(x, W) +b
"（4）结果分类"
#使用softmax进行分类化，把forward的特征转换成为分类的概率，结果赋给pred
pred = tf.nn.softmax(forward)
"""
3、训练模型
"""
"（1）设置训练参数"
train_epochs =5 #训练轮数
batch_size = 100 #单次训练样本数(批次大小)
total_batch= int(mnist. train. num_examples/batch_size)#一轮训练有多少批次
display_step =1 # 显示粒度,几轮一显示
learning_rate= 0.01 #学习率
"（2）定义损失函数"
loss_function=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
"（3）选择优化器"
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

"argmax的学习"
arr1 = np.array([1,3,2, 5, 7,0])
arr2 = np.array([[1.0,2,3],[3,2,1],[4,7,2],[8,3,21]])
print("arr1=",arr1)
print("arr2=\n",arr2)
argmax_1 = tf.argmax(arr1) #向量，不用指定第二个参数
argmax_20 = tf.argmax(arr2, 0) #指定第二个参数为0，按列方向搜索最大值
argmax_21 = tf.argmax(arr2, 1) #指定第二个参数为1，按行方向搜索最大值
argmax_22 = tf.argmax(arr2, -1) #指定第二个参数为-1，则第最后维的元素取值
# with tf. Session() as sess:
#     print(argmax_1.eval())
#     print(argmax_20.eval())
#     print(argmax_21.eval())
#     print(argmax_22.eval())

"（4）定义准确率"
#检查预测类别tf.argmax(pred, 1) 与实际类别tf.argmax(y, 1)的匹配情况
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf. reduce_mean(tf.cast(correct_prediction,tf.float32))
#声明会话并初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess. run(init)
"（5）模型训练"
#开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):#一轮分total_batch个批次
        xs,ys = mnist.train.next_batch(batch_size)# 1批次读batch_size个数据
        sess.run(optimizer, feed_dict={x: xs,y: ys}) #执行批次训练
    #total_batch个批次训练完成后，使用验证数据计算误差与准确率;验证集没有分批
    loss,acc= sess.run([loss_function, accuracy],
                       feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    #打印训练过程中的详细信息
    if (epoch+1) % display_step == 0:
        print("Train Epoch:","%02d"% (epoch+1),"Loss=","{:.9f}".format(loss),\
              "Accuracy=","{:.4f}".format(acc))
print("Train Finished!")

"""
4、评估模型
"""
#完成训练后，在测试集上评估模型的准确率，没有分批，相当于一次执行一万条数据
accu_test=sess.run(accuracy,
          feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy:", accu_test)

#完成训练后，在验证集上评估模型的准确率
accu_validation = sess.run(accuracy,
          feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
print("Test Accuracy:", accu_validation)

#完成训练后，在训练集上评估模型的准确率
accu_train = sess. run(accuracy,
            feed_dict={x:mnist.train.images,y:mnist.train.labels})
print("Test Accuracy:", accu_train)
"""
5、模型可视化
"""
#argmax能把pred值中最大值的下标给取出来，相当于转换为0~9数字
prediction_result=sess.run(tf.argmax(pred, 1),
                            feed_dict={x:mnist.test.images})
#查看前10项预测结果
print(prediction_result[0:10])


#定义可视化函数
import matplotlib. pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,#图像列表
                                  labels,#标签列表
                                  prediction,#预测值列表
                                  index,#从第index个开始显示
                                  num=10):#默认一次显示10幅
    fig = plt.gcf()  # 获取当前图表，Get Current Figure
    fig.set_size_inches(10, 12)  #设置图表大小，宽10英寸，高12英寸，1英寸等于2.54 cm
    if num > 25:
        num = 25# 最多显示25个子图
    for i in range(0,num):#它针对每一幅图像它是怎么处理
        ax = plt.subplot(5, 5, i + 1)  # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary')# 显示第index个图像
        title = "label="+str(np.argmax(labels[index])) #构建该图上要显示的title信息
        #if len(prediction) > 0:
        title += ",predict=" + str(prediction[index])#title上面显示预测值
        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()

plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,10,10)
