import os


def get_file(file_name):
    if not os.path.isdir(file_name):
        print("您输入的文件路径有误！！！")
    elif os.path.isdir(file_name):
        file_names = os.listdir(file_name)
        for name in file_names:
            # print(file_name)
            path = os.path.join(file_name, name)
            # print(file)
            if os.path.isfile(path):
                print(os.path.abspath(path))
            elif os.path.isdir(path):
                get_file(path)


if __name__ == "__main__":
    get_file(input("请输入您想查询的文件路径："))
