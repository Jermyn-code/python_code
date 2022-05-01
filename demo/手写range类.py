# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/30 0030 17:24
# @Deception：
# @Filename : 手写range类.py
class Range(object):
    def __init__(self, end, start=None, step=1):
        if end is not None:
            end, start = start, end
        if start is None:
            start = 0
        if step == 0:
            raise ValueError
        self.end = end
        self.start = start
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end & self.step < 0:
            raise StopIteration
        x = self.start
        self.start += self.step
        if self.start <= self.end:
            return x
        else:
            raise StopIteration


a = Range(2, 10, 3)
for i in a:
    print(i)
