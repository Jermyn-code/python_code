import jieba
from collections import Counter
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib.image as img

plt.rcParams['font.sans-serif'] = ["SimHei"]
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
fn = ""
wc = wordcount(font_path=fn, background_color="#74C7DA", random_state=100, max_font_size=150, width=610, hight=260)
w_clout = wc.generate(t)
s2 = plt.subplot(2, 1, 2)
s2.imshow(w_clout, interpolation="bilinear")
s2.axis("off")
plt.show()
