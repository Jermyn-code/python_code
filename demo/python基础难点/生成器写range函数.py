# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/5/1 0001 0:33
# @Deception：主要就是灵活使用 yield 的使用
# @Filename : 生成器写range函数.py
def _range(end, start=None, step=1):
    if start is None:
        start = 0
    if start is not None:
        end, start = start, end
    while start < end:
        yield start
        start += step


r = _range(2, 10, 2)
for i in r:
    print(i)
