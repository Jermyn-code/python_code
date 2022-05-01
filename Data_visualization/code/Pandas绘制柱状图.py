import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fn = "D:\\学习\\2022学期\\python_code\\Data_visualization\\covid(长春).xlsx"
    plt.rcParams['font.sans-serif'] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure("长春新冠疫情数据", figsize=(8, 5), layout="constrained")
    c = "区域"
    d = pd.read_excel(fn, sheet_name=None, index_col=c)
    i = 1
    for s in d.keys():
        p = plt.subplot(210 + i)
        plt.title("长春市主城区" + s + "分布图")
        aa = d[s].T.plot.bar(ax=p, stacked=True)
        aa.legend(loc=1, ncol=5, shadow=True)
        i += 1
    plt.show()
