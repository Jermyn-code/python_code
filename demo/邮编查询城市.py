code = input("请输入邮编：")
try:
    with open("youbian.txt", 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            city_code = eval(line[:-2])
            if str(city_code[0]) == code:
                print(city_code[1])
                break
        else:
            print("未找到相应的城市！！！！")
except FileNotFoundError:
    print("文件没有找到")
