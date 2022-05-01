import hashlib
# 生成MD5对象 并加盐
md5 = hashlib.md5(b'12345')
# 要加密的密码
password = '123456789'
# 对数据加密
md5.update(password.encode('utf-8'))
# 获取密文
pwd = md5.hexdigest()
print(pwd)

