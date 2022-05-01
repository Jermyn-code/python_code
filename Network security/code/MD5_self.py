# codeing=utf-8

# 引入math模块，因为要用到sin函数
import math

# 定义常量，用于初始化128位变量，注意字节顺序，文中的A=0x01234567，这里低值存放低字节，即01 23 45 67，所以运算时A=0x67452301，其他类似。
# 这里用字符串的形势，是为了和hex函数的输出统一，hex(10)输出为'0xA',注意结果为字符串。
A = '0x67452301'
B = '0xefcdab89'
C = '0x98badcfe'
D = '0x10325476'

# 定义每轮中用到的函数。L为循环左移，注意左移之后可能会超过32位，所以要和0xffffffff做与运算，确保结果为32位。
F = lambda x, y, z: ((x & y) | ((~x) & z))
G = lambda x, y, z: ((x & z) | (y & (~z)))
H = lambda x, y, z: (x ^ y ^ z)
I = lambda x, y, z: (y ^ (x | (~z)))
L = lambda x, n: (((x << n) | (x >> (32 - n))) & (0xffffffff))  # f相当于1111

# 定义每轮中循环左移的位数，这里用4个元组表示,用元组是因为速度比列表快。
shi_1 = (7, 12, 17, 22) * 4
shi_2 = (5, 9, 14, 20) * 4
shi_3 = (4, 11, 16, 23) * 4
shi_4 = (6, 10, 15, 21) * 4

# 定义每轮中用到的M[i]次序。
m_1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
m_2 = (1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12)
m_3 = (5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2)
m_4 = (0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9)


# 定义函数，用来产生常数T[i]，常数有可能超过32位，同样需要&0xffffffff操作。注意返回的是十进制的数。
def T(i):
    result = (int(4294967296 * abs(math.sin(i)))) & 0xffffffff
    return result


# 定义函数，用来将列表中的元素循环右移。原因是在每轮操作中，先运算A的值，然后是D，C，B，16轮之后右恢复原来顺序，所以只要每次操作第一个元素即可。
def shift(shift_list):
    shift_list = [shift_list[3], shift_list[0], shift_list[1], shift_list[2]]
    return shift_list


# 定义主要的函数，参数为当做种子的列表，每轮用到的F，G，H，I，生成的M[]，以及循环左移的位数。该函数完成一轮运算。
def fun(fun_list, f, m, shi):  # fun_list=[A,B,C，D]
    count = 0
    global Ti_count
    # 引入全局变量，T(i)是从1到64循环的。
    while count < 16:  # 16行运算
        # int(fun_list[0], 16)：代表将16进制转成整数
        # xx = (a + F(b,c,d) + Mj + ti) <<< s
        xx = int(fun_list[0], 16) + f(int(fun_list[1], 16), int(fun_list[2], 16), int(fun_list[3], 16)) + int(m[count],
                                                                                                              16) + T(
            Ti_count)
        # print("qian",xx)
        xx = xx & 0xffffffff
        # print("hou",xx)
        ll = L(xx, shi[count])
        # print("ll",ll)
        fun_list[0] = hex((int(fun_list[1], 16) + ll) & (0xffffffff))  # a = b+ xx<<<s
        fun_list = shift(fun_list)  # 循环右移一位
        count += 1
        Ti_count += 1
    return fun_list


# 该函数生成每轮需要的M[]，最后的参数是为了当有很多分组时，进行偏移。
def genM16(order, ascii_list, f_offset):
    ii = 0
    m16 = [0] * 16
    f_offset = f_offset * 64  # 当有第二个分组时候从第65个数字开始取
    for i in order:
        # print(order)
        i = i * 4  # 4个16进制数据合并为一个32位的一个分组：[0x6a,0x6b,0x6c,0x6d]-->[0x6a6b6c6d]
        m16[ii] = '0x' + ''.join((ascii_list[i + f_offset] + ascii_list[i + 1 + f_offset] + ascii_list[
            i + 2 + f_offset] + ascii_list[i + 3 + f_offset]).split('0x'))
        ii += 1
    print("M16前", m16)
    for c in m16:
        ind = m16.index(c)  # 获得其在list中的列表标号
        m16[ind] = reverse_hex(c)  # 进行翻转之后再赋值
    print("M16后", m16)
    return m16


# 翻转十六进制数的顺序：'0x01234567' => '0x67452301'
def reverse_hex(hex_str):
    hex_str = hex_str[2:]
    hex_str_list = []
    for i in range(0, len(hex_str), 2):
        hex_str_list.append(hex_str[i:i + 2])
    hex_str_list.reverse()
    hex_str_result = '0x' + ''.join(hex_str_list)
    return hex_str_result


# 显示结果函数，将最后运算的结果列表进行翻转，合并成字符串的操作。
def show_result(f_list):
    result = ''
    f_list1 = [0] * 4
    for i in f_list:
        f_list1[f_list.index(i)] = reverse_hex(i)[2:]
        result = result + f_list1[f_list.index(i)]
    return result


# 程序主循环
while True:
    abcd_list = [A, B, C, D]
    Ti_count = 1

    input_m = input('msg>>>')
    "1、将数字转为16进制"
    # 对每一个输入先添加一个'0x80'，即'10000000'
    ascii_list = list(map(hex, map(ord, input_m)))  # 转成16进制
    msg_lenth = len(ascii_list) * 8
    "2、对数据进行填充(填充1，填充0，填充长度)"
    # 补充1
    ascii_list.append('0x80')

    # 补充0
    while (len(ascii_list) * 8 + 64) % 512 != 0:
        ascii_list.append('0x00')

    # 补充长度
    # 例：'jklmn'，长度为'0x0800000000000000'，长度存放顺序低位在前
    msg_lenth_0x = hex(msg_lenth)[2:]  # 0x28
    msg_lenth_0x = '0x' + msg_lenth_0x.rjust(16, '0')  # 0x0000000000000028
    msg_lenth_0x_big_order = reverse_hex(msg_lenth_0x)[2:]  # 2800000000000000
    msg_lenth_0x_list = []
    for i in range(0, len(msg_lenth_0x_big_order), 2):  # 每两个数字进行一个切片
        msg_lenth_0x_list.append('0x' + msg_lenth_0x_big_order[i:i + 2])
        # msg_lenth_0x_list=['0x28', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00']
    ascii_list.extend(msg_lenth_0x_list)

    # ['0x6a', '0x6b', '0x6c', '0x6d', '0x6e', '0x80', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00',
    # '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00',
    # '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00',
    # '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00',
    # '0x28', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00']
    "3、"
    "（1）以512位（64个字节）为一块"
    # 对每个分组进行4轮运算
    for i in range(0, len(ascii_list) // 64):  # len(ascii_list)=64
        # 将最初128位种子存放在变量中，
        aa, bb, cc, dd = abcd_list
        "（2）分组"
        # 根据顺序产生每轮M[]列表
        order_1 = genM16(m_1, ascii_list, i)
        order_2 = genM16(m_2, ascii_list, i)
        order_3 = genM16(m_3, ascii_list, i)
        order_4 = genM16(m_4, ascii_list, i)
        "（3）FF、GG、HH、II 运算"
        # 主要四轮运算，注意打印结果列表已经被进行过右移操作！
        abcd_list = fun(abcd_list, F, order_1, shi_1)
        print("====================================")
        abcd_list = fun(abcd_list, G, order_2, shi_2)
        print("====================================")
        abcd_list = fun(abcd_list, H, order_3, shi_3)
        print("====================================")
        abcd_list = fun(abcd_list, I, order_4, shi_4)
        print("====================================")

        "(4) A = a + A; B = b + B; C = c + C; D= d + D"
        # 将最后输出与最初128位种子相加，注意，最初种子不能直接使用abcd_list[0]等，因为abcd_list已经被改变
        output_a = hex((int(abcd_list[0], 16) + int(aa, 16)) & 0xffffffff)
        output_b = hex((int(abcd_list[1], 16) + int(bb, 16)) & 0xffffffff)
        output_c = hex((int(abcd_list[2], 16) + int(cc, 16)) & 0xffffffff)
        output_d = hex((int(abcd_list[3], 16) + int(dd, 16)) & 0xffffffff)
        "（5）更新初始列表 ABCD"
        # 将输出放到列表中，作为下一次128位种子
        abcd_list = [output_a, output_b, output_c, output_d]
        print(abcd_list)
        # 将全局变量Ti_count恢复，一遍开始下一个分组的操作。
        Ti_count = 1

        # 最后调用函数，格式化输出
        print('md5>>>' + show_result(abcd_list))
