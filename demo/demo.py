# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/24 0024 12:04
# @Deception：
# @Filename : demo.py

def fibonacci(num):
    num1 = num2 = 1
    count = 0
    while count < num:
        count += 1
        yield num1
        num1, num2 = num2, num1 + num2


F = fibonacci(10)
for i in F:
    print(i)
