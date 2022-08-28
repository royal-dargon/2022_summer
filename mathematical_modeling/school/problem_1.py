import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import scipy.special as ss
from scipy.optimize import root
from scipy.stats import norm, gamma
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


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
    data_new = np.log10(data + 1.01)
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
    pca = PCA(n_components=0.95)
    x = pca.fit_transform(data)
    print(x.shape)
    # clustering = AgglomerativeClustering(n_clusters=15).fit(x)
    # clustering = KMeans(n_clusters=8).fit(x)
    # clustering = AffinityPropagation(random_state=2).fit(x)
    clustering = SpectralClustering(n_clusters=5).fit(x)
    labels = clustering.labels_
    labels_set = set(labels)
    number_labels = [[] for _ in range(len(labels_set))]
    nums_labels = [0 for _ in range(len(labels_set))]
    for i in range(len(labels)):
        number_labels[labels[i]].append(i)
        nums_labels[labels[i]] += 1
    print(len(number_labels))
    print(nums_labels)
    return number_labels


def calculate_weights(x, param):
    gauss = norm(loc=param[3], scale=param[4])
    pz_1 = param[0] * gamma.pdf(x, param[1], scale=1/param[2])
    pz_2 = (1 - param[0]) * gauss.pdf(x)
    pz = pz_1 / (pz_1 + pz_2)
    res = []
    # print(pz.shape, "pz_shape")
    for i in range(len(pz)):
        if np.isnan(pz[i]):
            pz[i] = 0
        res.append([pz[i], (1 - pz[i])])
    return np.array(res)


def calculate_w(x, param):
    gauss = norm(loc=param[3], scale=param[4])
    pz_1 = param[0] * gamma.pdf(x, param[1], scale=1/param[2])
    pz_2 = (1 - param[0]) * gauss.pdf(x)
    pz = pz_1 / (pz_1 + pz_2)
    return pz


def update_gmm_parameters(x, wt):
    tp_s = np.sum(wt)
    tp_t = np.sum(wt * x)
    tp_u = np.sum(wt * np.log(x))
    tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)
    # print(tp_s, tp_t, tp_u, tp_v)
    if tp_v <= 0:
        alpha = 20
    else:
        alpha0 = (3 - tp_v + np.sqrt((tp_v - 3)**2 + 24 * tp_v)) / 12 / tp_v
        # print(alpha0, "alpha0")
        if alpha0 >= 20:
            alpha = 20
        else:
            def fn(alpha_x):
                return np.log(alpha_x) - ss.digamma(alpha_x) - tp_v
            alpha = fsolve(fn, [0.9 * alpha0])
    beta = tp_s / tp_t * alpha
    return alpha, beta


def d_mix(x, param):
    gauss = norm(loc=param[3], scale=param[4])
    pz_1 = param[0] * gamma.pdf(x, param[1], scale=1/param[2])
    pz_2 = (1 - param[0]) * gauss.pdf(x)
    return pz_2 + pz_1


# 估计混合分布中的参数
def get_mix(x_data, point):
    # print(x_data, "x_data")
    init = [0 for _ in range(5)]
    x_data_rm = []
    for i in range(len(x_data)):
        if x_data[i] == point:
            init[0] += 1
        if x_data[i] > point:
            x_data_rm.append(x_data[i])
    # print(len(x_data_rm), init[0])
    init[0] = init[0] / len(x_data)
    if init[0] == 0:
        init[0] = 0.01
    init[1] = 0.5
    init[2] = 1
    x_data_rm = np.array(x_data_rm)
    init[3] = np.mean(x_data_rm)
    init[4] = np.std(x_data_rm)
    if np.isnan(init[3]):
        init[3] = 0
    if np.isnan(init[4]):
        init[4] = 0
    parameters = init
    eps = 10
    i_ter = 0
    log_old = 0
    # print(parameters)
    while eps > 0.5:
        # print(parameters, i_ter+1)
        w_t = calculate_weights(x_data, parameters)
        w_t = w_t.reshape(-1, 2)
        # print(w_t.shape, "wt_shape")
        parameters[0] = np.sum(w_t[:, 0]) / len(w_t)
        print(parameters[0])
        parameters[3] = np.sum(w_t[:, 1] * x_data) / np.sum(w_t[:, 1])
        parameters[4] = np.sqrt(np.sum(w_t[:, 1] * (x_data - parameters[3])**2) / np.sum(w_t[:, 1]))
        parameters[1], parameters[2] = update_gmm_parameters(x=x_data, wt=w_t[:, 0])

        log_lik = np.sum(np.log10(d_mix(x_data, parameters)))
        eps = (log_lik - log_old)**2
        log_old = log_lik
        i_ter += 1
        if i_ter > 50:
            break
    print(parameters)
    return parameters


def get_mix_parameters(count, point):
    parameters = []
    print(count.shape, "count")
    for i in range(count.shape[1]):
        x_data = count[:, i]
        # print(x_data.shape, i)
        param = get_mix(x_data, point)
        parameters.append(param)
    return parameters


def main():
    data_real = get_data("data/real_data1.csv")
    data1 = get_data("data/data1.csv")
    labels = my_cluster(data1)
    # print(labels[15][0])
    # print(data[624])
    x_data = []
    # data = np.log10(data + 0.01)
    data = data_format(data1)
    for i in range(len(labels)):
        x = []
        for n in range(len(labels[i])):
            x.append(data[labels[i][n]])
        x = np.array(x)
        x_data.append(x)
    points = np.log10(1.01)
    label_parameters = []
    print(x_data[0].shape)
    for i in range(len(x_data)):
        res = get_mix_parameters(x_data[i], points)
        label_parameters.append(res)
    print(len(label_parameters))
    sum_num = 0
    correct = 0
    for i in range(len(x_data)):
        # print(labels[i])
        # print(x_data[i][0, :17])
        for n in range(x_data[i].shape[1]):
            d = calculate_w(x_data[i][:, n], label_parameters[i][n])
            m = np.mean(d)
            print(m, "m")
            for j, p in enumerate(d):
                if data1[labels[i][j]][n] == 0:
                    if p > m and data_real[labels[i][j]][n] != 0:
                        correct += 1
                        sum_num += 1
                    elif p < m and data_real[labels[i][j]][n] != 0:
                        sum_num += 1
    print(correct, sum_num, "res")
    print(correct/sum_num)


if __name__ == "__main__":
    main()

