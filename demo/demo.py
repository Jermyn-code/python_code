# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/24 0024 12:04
# @Deception：
# @Filename : demo.py

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