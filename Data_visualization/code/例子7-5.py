# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/5/1 0001 13:14
# @Deception：编写绘制y=cosX的切线沿曲线运动的动画
# @Filename : 例子7-5.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
plt.grid(ls='--')
x = np.linspace(0.2 * np.pi, 100)
y = np.sin(x)

carve_ani = plt.plot(x, y, 'red', alpha=0.5)[0]

point_ani = plt.plot(0, 0, 'g', alpha=0.4, marker='o')[0]

xtext_ani = plt.text(5, 0.8, '', frontsize=12)
ytext_ani = plt.text(5, 0.7, '', frontsize=12)
ktext_ani = plt.text(5, 0.6, '', frontsize=12)


def tangent_line(x0, y0, k):
    xs = np.linespace(x0 - 0.5, x0 + 0.5, 100)
    ys = y0 + k * (xs - x0)
    return xs, ys


def slope(x0):
    num_min = np.sin(x0 - 0.05)
    num_max = np.sin(x0 + 0.05)
    k = (num_max - num_min) / 0.1
    return k


k = slope(x[0])
xs, ys = tangent_line(x[0], y[0], k)

tangent_ani = plt.plot(xs, ys, c='blue', alpha=0.8)[0]


# 更新数据

def updata(num):
    k = slope(x[num])
    xs, ys = tangent_line(x[num], y[num], k)
    tangent_ani.set_data(xs, ys)
    point_ani.set_text('x=%.3f' % x[num])
    point_ani.set_text('x=%.3f' % y[num])
    ktext_ani.set_text('k=%.3f' % k)
    return [point_ani, xtext_ani, ytext_ani, tangent_ani, k]


ani = animation.FuncAnimation(fig=fig, func=updata, fargs=np.arange(0, 100), interval=100)
plt.show()
