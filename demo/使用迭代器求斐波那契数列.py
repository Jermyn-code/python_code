# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/30 0030 22:36
# @Deception：优点，使用的迭代器求不存数组，相当于拿时间换空间，普通方法求不得的巨大数，此方法可使用
# @Filename : 使用迭代器求斐波那契数列.py
class Fibonacci(object):
    def __init__(self, num):
        self.num = num
        self.num1 = self.num2 = 1
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # 每次调用 __next__ 方法，次数就加一
        self.count += 1
        # 定义一个变量，用来保存修改以前的num1值
        x = self.num1
        self.num1, self.num2 = self.num2, self.num1 + self.num2
        if self.count <= self.num:
            return x
        else:
            raise StopIteration


fib = Fibonacci(15000)
for x in fib:
    print(x)
