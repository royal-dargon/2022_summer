import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.stats import pearsonr



# this is the names of data
data_name = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
# this is the temperature of the tests(the difference should be noticed in test A1, A2, A3...)
temperatures_1to2 = [250, 275, 300, 325, 350]
temperatures_3 = [250, 275, 300, 325, 350, 400, 450]
temperatures_4to5 = [250, 275, 300, 325, 350, 400]
temperatures_6to16 = [250, 275, 300, 350, 400]
temperatures_17to21 = [250, 275, 300, 325, 350, 400]
# the name of excel2's list
excel_2_name = ['乙醇转化率(%)', '乙烯选择性', 'C4烯烃选择性', '乙醛选择性', '碳数为4-12脂肪醇', '甲基苯甲醛和甲基苯甲醇']


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


# get data from excel2
def get_data_2():
    data = pd.read_excel("data2.xlsx", skiprows=[0, 1, 2], header=None)
    list_time = []
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    for i in range(len(data)):
        list_time.append(data.iloc[i, 0])
        list_1.append(data.iloc[i, 1])
        list_2.append(data.iloc[i, 2])
        list_3.append(data.iloc[i, 3])
        list_4.append(data.iloc[i, 4])
        list_5.append(data.iloc[i, 5])
        list_6.append(data.iloc[i, 6])
    parameters = {
        "list1": list_time,
        "list2": list_1,
        "list3": list_2,
        "list4": list_3,
        "list5": list_4,
        "list6": list_5,
        "list7": list_6
    }
    return data, parameters


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
    plt.xlabel("温度 （℃）")
    plt.ylabel("乙醇转化率 （%）")
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
    plt.xlabel("温度 （℃）")
    plt.ylabel("C4烯烃选择性 （%）")
    plt.title("C4烯烃选择性与温度的关系")
    plt.legend()
    plt.show()


# fit my data to the func
def fit_data(all_eth, all_c4):
    n = len(all_eth)
    predict_data = [i for i in range(200, 451, 10)]
    predict_data = np.array(predict_data).reshape(-1, 1)
    degree = 2
    # plt.subplots_adjust(wspace=0.8, hspace=0.8)
    for i in range(n):
        # degree代表最高次数
        clf_1 = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                        ('linear', LinearRegression(fit_intercept=False))])
        clf_2 = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                        ('linear', LinearRegression(fit_intercept=False))])
        if i < 2:
            clf_1.fit(np.array(temperatures_1to2).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre_1 = clf_1.predict(predict_data)
            plt.subplot(1, 2, 1)
            plt.scatter(temperatures_1to2, all_eth[i], c='red')
            plt.plot(predict_data, y_pre_1, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("乙醇转化率 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " 乙醇转化率与温度的关系", fontsize=10)
            clf_2.fit(np.array(temperatures_1to2).reshape(-1, 1), np.array(all_c4[i]).reshape(-1, 1))
            y_pre_2 = clf_2.predict(predict_data)
            plt.subplot(1, 2, 2)
            plt.scatter(temperatures_1to2, all_c4[i], c='red')
            plt.plot(predict_data, y_pre_2, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("C4烯烃选择性 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
            plt.show()
        elif i < 3:
            clf_1.fit(np.array(temperatures_3).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre_1 = clf_1.predict(predict_data)
            plt.subplot(1, 2, 1)
            plt.scatter(temperatures_3, all_eth[i], c='red')
            plt.plot(predict_data, y_pre_1, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("乙醇转化率 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " 乙醇转化率与温度的关系", fontsize=10)
            clf_2.fit(np.array(temperatures_3).reshape(-1, 1), np.array(all_c4[i]).reshape(-1, 1))
            y_pre_2 = clf_2.predict(predict_data)
            plt.subplot(1, 2, 2)
            plt.scatter(temperatures_3, all_c4[i], c='red')
            plt.plot(predict_data, y_pre_2, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("C4烯烃选择性 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
            plt.show()
        elif i < 5:
            clf_1.fit(np.array(temperatures_4to5).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre_1 = clf_1.predict(predict_data)
            plt.subplot(1, 2, 1)
            plt.scatter(temperatures_4to5, all_eth[i], c='red')
            plt.plot(predict_data, y_pre_1, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("乙醇转化率 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " 乙醇转化率与温度的关系", fontsize=10)
            clf_2.fit(np.array(temperatures_4to5).reshape(-1, 1), np.array(all_c4[i]).reshape(-1, 1))
            y_pre_2 = clf_2.predict(predict_data)
            plt.subplot(1, 2, 2)
            plt.scatter(temperatures_4to5, all_c4[i], c='red')
            plt.plot(predict_data, y_pre_2, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("C4烯烃选择性 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
            plt.show()
        elif i < 16:
            clf_1.fit(np.array(temperatures_6to16).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre_1 = clf_1.predict(predict_data)
            plt.subplot(1, 2, 1)
            plt.scatter(temperatures_6to16, all_eth[i], c='red')
            plt.plot(predict_data, y_pre_1, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("乙醇转化率 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " 乙醇转化率与温度的关系", fontsize=10)
            clf_2.fit(np.array(temperatures_6to16).reshape(-1, 1), np.array(all_c4[i]).reshape(-1, 1))
            y_pre_2 = clf_2.predict(predict_data)
            plt.subplot(1, 2, 2)
            plt.scatter(temperatures_6to16, all_c4[i], c='red')
            plt.plot(predict_data, y_pre_2, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("C4烯烃选择性 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
            plt.show()
        elif i < 21:
            clf_1.fit(np.array(temperatures_17to21).reshape(-1, 1), np.array(all_eth[i]).reshape(-1, 1))
            y_pre_1 = clf_1.predict(predict_data)
            plt.subplot(1, 2, 1)
            plt.scatter(temperatures_17to21, all_eth[i], c='red')
            plt.plot(predict_data, y_pre_1, c='blue')
            plt.xlabel("温度（℃）", fontsize=10)
            plt.ylabel("乙醇转化率 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " 乙醇转化率与温度的关系", fontsize=10)
            clf_2.fit(np.array(temperatures_17to21).reshape(-1, 1), np.array(all_c4[i]).reshape(-1, 1))
            y_pre_2 = clf_2.predict(predict_data)
            plt.subplot(1, 2, 2)
            plt.scatter(temperatures_17to21, all_c4[i], c='red')
            plt.plot(predict_data, y_pre_2, c='blue')
            plt.xlabel("温度 （℃）", fontsize=10)
            plt.ylabel("C4烯烃选择性 （%）", fontsize=10)
            plt.title(str(data_name[i]) + " C4烯烃选择性与温度的关系", fontsize=10)
            plt.show()


# 手写皮尔威系数计算
def ppw(x, y):
    n = len(x)
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pcc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pcc


# 计算皮尔威系数
def my_ppw(parameters):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    time_list = parameters['list1']
    all_list = []
    all_list.append(parameters['list2'])
    all_list.append(parameters['list3'])
    all_list.append(parameters['list4'])
    all_list.append(parameters['list5'])
    all_list.append(parameters['list6'])
    all_list.append(parameters['list7'])
    for i in range(len(all_list)):
        for n in range(i+1, len(all_list)):
            # num = ppw(np.array(all_list[i]), np.array(all_list[n]))
            # res = np.corrcoef(all_list[i], all_list[n])
            # res = pearsonr(all_list[i], all_list[n])
            # print(str(excel_2_name[i]) + '与' + str(excel_2_name[n]) + '的皮威尔系数为 ' + str(num) + '。置信区间为' + str(res))
            plt.scatter(all_list[i], all_list[n])
            plt.xlabel(excel_2_name[i])
            plt.ylabel(excel_2_name[n])
            plt.show()


def main():
    # all_eth, all_c4 = get_data()
    # print_plot(all_eth, all_c4)
    # fit_data(all_eth, all_c4)
    data2, parameters = get_data_2()
    my_ppw(parameters)


if __name__ == "__main__":
    main()

