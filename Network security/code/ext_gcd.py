def ext_gcd(a, b):
    if b == 0:
        return 1, 0, a
    else:
        x, y, q = ext_gcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, q


#产生秘钥d
def generate_d(phn, e):
    (x, y, r) = ext_gcd(phn, e)
    # y maybe < 0, so convert it
    if y < 0:
        print(y)
        #return y % ph_n
        return y + phn  #直接用加法效率高一丢丢
    return y
d = generate_d(3120, 17)
print(d)
