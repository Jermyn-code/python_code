#!/usr/bin/env python
# coding -*- utf:8 -*-
import math
import random


# 生成素数数组
def prime_array():
    arraya = []
    for i in range(2, 100):  # 生成前100中的素数，从2开始因为2是最小的素数
        x = is_prime(i)  # i为素数是返回True，则将x加入arraya数组中;2为测试值
        if x:
            arraya.append(i)
    return arraya


# 判断是否是素数
def is_prime(n):
    if n == 2:
        return True
    for i in range(2, n // 2 + 1):
        if n % i == 0:
            return False
    return True


# 找出与（p-1）*(q-1)互质的数e
def co_prime(phn):
    while True:
        e = random.choice(range(10, 1000))
        x = gcd(e, phn)
        if x == 1:  # 如果最大公约数为1，则退出循环返回e
            break
    return e


# 求两个数的最大公约数
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


# 根据e*d mod phn = 1,找出d
def find_d(e, phn):
    for d in range(100000000):  # 随机太难找，就按顺序找到d,range里的数字随意
        x = (e * d) % phn
        if x == 1:
            return d


"""
运用了扩展欧几里的算法，求a*x + b*y = 1
递归条件:当m==0时，gcd(n,m)=n，此时x=1,y=0
"""


def ext_gcd(a, b):
    if b == 0:
        return 1, 0, a
    else:
        x, y, q = ext_gcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, q


# 产生秘钥d
def generate_d(phn, e):
    (x, y, r) = ext_gcd(phn, e)
    # y maybe < 0, so convert it
    if y < 0:
        print(y)
        # return y % ph_n
        return y + phn  # 直接用加法效率高一丢丢
    return y


# 生成公钥和私钥
def main():
    a = prime_array()
    print("前100个素数:", a)
    p = random.choice(a)
    q = random.choice(a)
    print("随机生成两个素数p和q. p=", p, " q=", q)
    n = p * q
    phn = (p - 1) * (q - 1)
    e = co_prime(phn)
    print("根据e和(p-1)*(q-1))互质得到: e=", e)
    d = generate_d(phn, e)
    d = find_d(e, phn)
    print("根据(e*d) 模 ((p-1)*(q-1)) 等于 1 得到 d=", d)
    print("公钥:   n=", n, "  e=", e)
    print("私钥:   n=", n, "  d=", d)
    pbvk = (n, e, d)
    return pbvk


# 生成public key公钥或private key私钥
# zx==0 公钥 zx==1 私钥
# a为元组(n,e,d)
def generate_pbk_pvk(a, zx):
    pbk = (a[0], a[1])  # public key公钥 元组类型，不能被修改
    pvk = (a[0], a[2])  # private key私钥
    # print("公钥:   n=",pbk[0],"  e=",pbk[1])
    # print("私钥:   n=",pvk[0],"  d=",pvk[1])
    if zx == 0:
        return pbk
    if zx == 1:
        return pvk


# 加密
def encryption(mw, ned):
    # 密文B = 明文A的e次方 模 n， ned为公钥
    # mw就是明文A，ned【1】是e， ned【0】是n
    B = pow(mw, ned[1]) % ned[0]
    return B


# 解密
def decode(mw, ned):
    # 明文C = 密文B的d次方 模 n， ned为私钥匙
    # mw就是密文B， ned【1】是e，ned【1】是d
    C = pow(mw, ned[1]) % ned[0]
    return C


def en_rsa(pbvk, A):
    pbk = generate_pbk_pvk(pbvk, 0)  # 公钥  if 0 return pbk if 1 return pvk
    B = [encryption(int(a), pbk) for a in A]  # 加密
    print(B)
    return B


def de_rsa(pbvk, B):
    pvk = generate_pbk_pvk(pbvk, 1)  # 私钥
    C = [decode(b, pvk) for b in B]
    print(C)
    return C


if __name__ == '__main__':
    pbvk = main()
    print("请输入明文:", end='')
    A = [ord(i) for i in list(input())]
    B = en_rsa(pbvk, A)
    C = de_rsa(pbvk, B)
    print("解密后的结果:", end='')
    for j in C:
        print(chr(int(j)), end='')
