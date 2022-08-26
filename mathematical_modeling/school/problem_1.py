import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# this is the func of read the data
def get_data(file_name):
    data = pd.read_csv(file_name)
    print(data.shape)
    data_list = []
    for i in range(data.shape[1]):
        data_list.append(data.iloc[:, i])
    return np.array(data_list).astype(float)


# 这是进行归一化操作的函数
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# this is the func of format the data
def data_format(data):
    data = normalization(data)
    data_new = np.log10(data + 0.01)
    return data_new


# 绘制类别分类的柱形图
def kind_bar(data):
    n = len(data)
    name_labels = []
    for i in range(n):
        s = "类别" + str(i+1)
        name_labels.append(s)
    plt.bar(range(n), data, tick_label=name_labels, width=0.7)
    plt.show()


# 对数据进行聚类
def my_cluster(data):
    data = data_format(data)
    pca = PCA(n_components=0.7)
    x = pca.fit_transform(data)
    print(x.shape)
    clustering = AffinityPropagation(random_state=2).fit(x)
    labels = clustering.labels_
    labels_set = set(labels)
    number_labels = [[] for _ in range(len(labels_set))]
    nums_labels = [0 for _ in range(len(labels_set))]
    for i in range(len(labels)):
        number_labels[labels[i]].append(i)
        nums_labels[labels[i]] += 1
    print(nums_labels)


def main():
    data = get_data("data/data1.csv")
    my_cluster(data)


if __name__ == "__main__":
    main()

