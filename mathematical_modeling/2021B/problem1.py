import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np


# this is the names of data
data_name = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
# this is the temperature of the tests(the difference should be noticed in test A1, A2, A3...)
temperatures_1to2 = [250, 275, 300, 325, 350]
temperatures_3 = [250, 275, 300, 325, 350, 400, 450]
temperatures_4to5 = [250, 275, 300, 325, 350, 400]
temperatures_6to16 = [250, 275, 300, 350, 400]
temperatures_17to21 = [250, 275, 300, 325, 350, 400]


# get data from the excel
def get_data():
    data = pd.read_excel("data1.xlsx")
    all_eth = []
    all_c4 = []
    i = 0
    for n in range(21):
        eth_conversion = []
        c4 = []
        eth_conversion.append(data.iloc[i, 3])
        c4.append(data.iloc[i, 5])
        i = i + 1
        while i < 114 and str(data.iloc[i, 0]) == 'nan':
            eth_conversion.append(data.iloc[i, 3])
            c4.append(data.iloc[i, 5])
            i = i + 1
        all_eth.append(eth_conversion)
        all_c4.append(c4)
    return all_eth, all_c4


# this is the func of drawing the pictures about data
def print_plot(all_eth, all_c4):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    for i in range(len(all_eth)):
        if i < 2:
            plt.plot(temperatures_1to2, all_eth[i], label=data_name[i])
            # plt.plot(temperatures_1to2, all_c4[i], label=data_name[i])
        elif i < 3:
            plt.plot(temperatures_3, all_eth[i], label=data_name[i])
        elif i < 5:
            plt.plot(temperatures_4to5, all_eth[i], label=data_name[i])
        elif i < 16:
            plt.plot(temperatures_6to16, all_eth[i], label=data_name[i])
        elif i < 21:
            plt.plot(temperatures_17to21, all_eth[i], label=data_name[i])
    plt.xlabel("温度")
    plt.ylabel("乙醇转化率")
    plt.title("乙醇转化率与温度的关系")
    plt.subplot(1, 2, 2)
    for i in range(len(all_eth)):
        if i < 2:
            plt.plot(temperatures_1to2, all_c4[i], label=data_name[i])
            # plt.plot(temperatures_1to2, all_c4[i], label=data_name[i])
        elif i < 3:
            plt.plot(temperatures_3, all_c4[i], label=data_name[i])
        elif i < 5:
            plt.plot(temperatures_4to5, all_c4[i], label=data_name[i])
        elif i < 16:
            plt.plot(temperatures_6to16, all_c4[i], label=data_name[i])
        elif i < 21:
            plt.plot(temperatures_17to21, all_c4[i], label=data_name[i])
    plt.xlabel("温度")
    plt.ylabel("C4烯烃选择性")
    plt.title("C4烯烃选择性与温度的关系")
    plt.legend()
    plt.show()


# fit my data to the func
def fit_data(all_eth, all_c4):
    n = len(all_eth)
    predict_data = [i for i in range(200, 451, 10)]
    predict_data = np.array(predict_data).reshape(-1, 1)
    degree = 2
    plt.subplots_adjust(wspace=0.8, hspace=0.8)  # 调整子图间距
    for i in range(n):
        # degree代表最高次数
        clf = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                        ('linear', LinearRegression(fit_intercept=False))])
        if i < 2:
            clf.fit(np.array(temperatures_1to2).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre = clf.predict(predict_data)
            plt.subplot(5, 5, i+1)
            plt.scatter(temperatures_1to2, all_eth[i], c='red')
            plt.plot(predict_data, y_pre, c='blue')
            plt.xlabel("温度", fontsize=10)
            plt.ylabel("乙醇转化率", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
        elif i < 3:
            clf.fit(np.array(temperatures_3).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre = clf.predict(predict_data)
            plt.subplot(5, 5, i + 1)
            plt.scatter(temperatures_3, all_eth[i], c='red')
            plt.plot(predict_data, y_pre, c='blue')
            plt.xlabel("温度", fontsize=10)
            plt.ylabel("乙醇转化率", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
        elif i < 5:
            clf.fit(np.array(temperatures_4to5).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre = clf.predict(predict_data)
            plt.subplot(5, 5, i + 1)
            plt.scatter(temperatures_4to5, all_eth[i], c='red')
            plt.plot(predict_data, y_pre, c='blue')
            plt.xlabel("温度", fontsize=10)
            plt.ylabel("乙醇转化率", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
        elif i < 16:
            clf.fit(np.array(temperatures_6to16).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre = clf.predict(predict_data)
            plt.subplot(5, 5, i + 1)
            plt.scatter(temperatures_6to16, all_eth[i], c='red')
            plt.plot(predict_data, y_pre, c='blue')
            plt.xlabel("温度", fontsize=10)
            plt.ylabel("乙醇转化率", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
        elif i < 21:
            clf.fit(np.array(temperatures_17to21).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre = clf.predict(predict_data)
            plt.subplot(5, 5, i + 1)
            plt.scatter(temperatures_17to21, all_eth[i], c='red')
            plt.plot(predict_data, y_pre, c='blue')
            plt.xlabel("温度", fontsize=10)
            plt.ylabel("乙醇转化率", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
    plt.show()


def main():
    all_eth, all_c4 = get_data()
    print_plot(all_eth, all_c4)
    fit_data(all_eth, all_c4)


if __name__ == "__main__":
    main()

