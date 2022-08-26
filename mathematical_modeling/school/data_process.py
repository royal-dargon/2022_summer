import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# this is the func of format the data
def get_data(file_name):
    data = pd.read_csv(file_name)
    print(data.shape)
    data_list = []
    for i in range(data.shape[1]):
        data_list.append(data.iloc[:, i])
    return data_list


# this is the func of get label
def get_label(file_name):
    data = pd.read_excel(file_name, header=None)
    data_list = []
    for i in range(len(data)):
        data_list.append(data.iloc[i, 0])
    return data_list


# 这是对错误的基因组的个数进行统计
def incorrect_gene(file_name_1, file_name_2, file_name_3):
    data = get_data(file_name_1)
    label = get_data(file_name_2)
    name = get_label(file_name_3)
    name_num = {}
    for i in range(len(name)):
        name_num[name[i]] = 0
    print(len(label))
    print(len(label[1]))
    for i in range(len(label[0])):
        for n in range(len(label)):
            if label[n][i] == 1:
                name_num[name[i]] += 1
    keys = []
    values = []
    for key, value in name_num.items():
        keys.append(key)
        values.append(value)
    plt.bar(range(len(keys)), values, tick_label=keys)
    plt.show()


# 每个细胞出现drop out次数的估计
def incorrect_cell(file_name_1, file_name_2):
    label = get_data(file_name_1)
    name = get_label(file_name_2)
    res = [0 for i in range(len(label))]
    for i in range(len(label)):
        for n in range(len(label[i])):
            if label[i][n] == 1:
                res[i] += 1
    print(len(res))
    print(res)


def main():
    # incorrect_gene("data/data1.csv", "data/label_data1.csv", "data/name1.xlsx")
    incorrect_cell("data/label_data1.csv", "data/name1.xlsx")


if __name__ == "__main__":
    main()
