from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# 生成 shape(n,)的随机数组
def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


# 设置每组样式和范围
# x在[23,32]，y在[0,100].z在[zlow,zhigh]范围生成随机点
# 将两组散点值绘制到同一个 figure中
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '*', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.set_zlabel("z label")
    plt.show()
