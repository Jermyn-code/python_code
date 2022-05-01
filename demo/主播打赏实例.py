# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/23 0023 18:44
# @Deception:对主播进行打赏，金额写入对应的csv文件中
# @Filename : 主播打赏实例.py


import csv, os

if not os.path.exists('anchor.csv'):
    f = open('anchor.csv', 'w', encoding='utf8')
    f.write('主播,金额\n')
    f.close()
csv_w = open('anchor.csv', 'a', encoding='utf8', newline='')
csv_r = open('anchor.csv', 'r', encoding='utf8')
file_r = csv.reader(csv_r)
user_dict = {}
while True:
    name = input("请输入主播名字：")
    if name == 'exit':
        break
    try:
        money = float(input("请输入打赏金额："))
    except ValueError as v:
        print("您输入的金额有误")
        break
    file_w = csv.writer(csv_w)
    file_w.writerow([name, money])

# -*- coding: utf-8 -*-
# @Author : Jermyn
# @Time : 2022/4/23 0023 20:52
# @Deception：计算主播的打赏总金额
# @Filename : demo2.py
# import csv
#
# csv_r = open('anchor.csv', 'r', encoding='utf8')
#
# file_r = csv.reader(csv_r)
#
# user_dict = {}
# for name, money in file_r:
#     if name != '主播':
#         user_dict[name] = user_dict.get(name, 0) + float(money)
#     #     user_dict[name] = float(money)
#     # else:
#     #     user_dict[name] += float(money)
# for name, money in user_dict:
#     user_dict[name] = user_dict.get(name, 0) + float(money)
# csv_r.close()
# for k, v in user_dict.items():
#     print(f"{k} 总收入:{v}")
