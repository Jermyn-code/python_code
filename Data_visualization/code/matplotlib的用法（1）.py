import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 2, 100)
y = np.array([35, 25, 25, 15])
x_bar = [0.3, 1.7, 4, 6, 7]
y_bar = [5, 20, 15, 25, 10]
fig, axs = plt.subplots(2, 2, figsize=(6, 9), layout="constrained")

plt.rcParams['font.sans-serif'] = ["SimHei"]
axs[0, 0].plot(x, x, label='linear')
axs[0, 0].set_title("Simple y=x")
axs[0, 1].pie(y, labels=['A', 'B', 'C', 'D'])
axs[0, 1].legend()
axs[1, 0].bar(x_bar, y_bar, width=0.6, bottom=[10, 0, 5, 0, 5], color='green')
axs[1, 1].plot(x, x ** 2, label='quadratic')
axs[1, 1].set_title('Simple y=x^2')
fig.show()
