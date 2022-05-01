# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/23 0023 17:39
# @Deception：python自带库遍历获取文件夹下所有文件
# @File : python自带库遍历获取文件夹下所有文件.py
import os


def get_file2(path):
    for path, dirs, files in os.walk(path):

        for file in files:
            print(os.path.abspath(os.path.join(path, file)))


if __name__ == "__main__":
    get_file2(input("请输入您想查询的文件路径："))
