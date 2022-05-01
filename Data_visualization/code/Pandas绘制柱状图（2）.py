import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from collections import Counter
from wordcount import wordcount



plt.rcParams['font.sans-serif'] = ["SimHei"]


def DrawBarinPyplot(data):
    n_rows = len(data)
    colors = ["#CC4D3C", "#4ABE73", "#280EE8", "#30DCD8", "#A133C3", "#B4B442", "#97605F", "#D8E80E",
              "#D620DA", "#7C0679", "#76916D", "#8E6B68", "#B6406A", "#1DEB53", "#4B4344"]
    index = data.columns.values
    y_offset = np.zeros(len(data.columns))
    for row in range(n_rows):
        a = data.index[row]
        plt.bar(index, data.values[row], 0.4,
                bottom=y_offset, color=colors[row], label=a)
        y_offset += data.values[row]
    plt.ylabel("当日各城区确诊病例总数")
    plt.xlabel("2022年4月各日历日")
    plt.legend(loc=1, ncol=5, shadow=True)
    plt.title("长春市主城区确诊病例分布图")
    plt.show()


def WordCloud():
    plt.figure("长春抗疫", figsize=(7, 6), layout="constrained")
    s1 = plt.subplot(2, 1, 1)
    s1.axis('off')
    fn = 'D:\\学习\\2022学期\\python_code\\Data_visualization\\code\\wordcloud\\'
    image = img.imread(fn)
    s1.imshow(image)
    fn = "D:\\学习\\2022学期\\python_code\\Data_visualization\\code\\wordcloud\\"
    text = open(fn, encoding="utf-8").read()
    doc_data = text
    jieba.add_word("新冠疫情")
    words = jieba.lcut(doc_data)
    words_list = []
    words_list = [w for w in words if len(w) > 1]
    word_c = Counter(words_list)
    top = word_c.most_common(160)
    t = " "
    for w in top: t += w[0] + " "
    fn = " "
    wc = wordcount(font_path=fn, background_color="#74C7DA", random_state=100, max_font_size=150, width=610, hight=260)
    w_clout = wc.generate(t)
    s2 = plt.subplot(2, 1, 2)
    s2.imshow(w_clout, interpolation="bilinear")
    s2.axis("off")
    plt.show()


if __name__ == " main ":
    fn = "D:\\学习\\2022学期\\python_code\\Data_visualization\\covid(长春).xlsx"
    plt.rcParams['font.sans-serif'] = ["SimHei"]
    plt.figure("长春新冠疫情数据", fiqsize=(8, 5))
    c = "区域"
    d = pd.read_excel(fn, sheet_name="确诊病例", indexcol=c)
    DrawBarinPyplot(d)
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
    WordCloud()
    plt.show()
