import matplotlib.pyplot as plt

import numpy as np

plt.figure("实例1", figsize=(8, 6), facecolor='#aaeeff', edgecolor='#FF0000', frameon=True)
plt.subplot(2, 1, 1)
x = np.arange(0, 2 * np.pi, 0.1)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, marker=".", linewidth=1, linestyle="--", color="red")
plt.plot(x, z)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.title("三角函数曲线示例 y=sinx , z=cosx")
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.legend(["Y", "z"], loc="upper right")
plt.show()
