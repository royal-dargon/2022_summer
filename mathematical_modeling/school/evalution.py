import pandas as pd
import numpy as np


def get_data(file_name):
    data = pd.read_csv(file_name)
    # print(data.shape, "1")
    data_list = []
    for i in range(data.shape[1]):
        data_list.append(data.iloc[:, i])
    return np.array(data_list).astype(float)


def get_res(file_name):
    data = pd.read_csv(file_name)
    # print(data.shape, "1")
    data_list = []
    for i in range(1, data.shape[1]):
        data_list.append(data.iloc[:, i])
    return np.array(data_list).astype(float)


def value2out(file_1, file_2):
    data_1 = get_data(file_1)
    data_2 = get_res(file_2)
    drop_dict = {}
    white = []
    for i in range(data_1.shape[1]):
        name = "gene" + str(i+1)
        white.append(name)
    drop_dict[" "] = white
    for i in range(data_1.shape[0]):
        d_l = []
        for n in range(data_1.shape[1]):
            if data_1[i, n] == 0:
                if data_2[i, n] != 0:
                    d_l.append(1)
                else:
                    d_l.append(-1)
            else:
                d_l.append(0)
        name = "cell" + str(i+1)
        drop_dict[name] = d_l
    df = pd.DataFrame(drop_dict)
    df.to_csv("drop_res1.csv", index=False, sep=',')


def evaluate_problem_1(file_1, file_2):
    data_1 = get_data(file_1)
    data_2 = get_res(file_2)
    correct = 0
    sum_num = 0
    for i in range(data_1.shape[0]):
        for n in range(data_1.shape[1]):
            if data_1[i, n] == 1:
                if data_2[i, n] == 1:
                    correct += 1
                    sum_num += 1
                else:
                    sum_num += 1
    print(correct, sum_num, correct/sum_num)


def evaluate_problem_2(file_1, file_2, file_3, file_4):
    data_1 = get_data(file_1)
    # label
    data_2 = get_data(file_2)
    # real
    data_3 = get_data(file_3)
    data_4 = get_res(file_4)
    num_drop = 0
    sum_f = 0
    for i in range(data_1.shape[0]):
        for n in range(data_1.shape[1]):
            if data_2[i, n] == 1:
                num_drop += 1
                f = ((data_4[i, n] - data_3[i, n])**2) / ((data_1[i, n] - data_3[i, n])**2)
                sum_f += f
    print(sum_f, num_drop)
    res = sum_f / num_drop
    print(res)


def main():
    evaluate_problem_1("data/label_data1.csv", "data/res_name1_20.csv")
    evaluate_problem_2("data/data1.csv", "data/label_data1.csv", "data/real_data1.csv", "data/res1_20.csv")
    # value2out("data/data1.csv", "data/res1_10.csv")


if __name__ == "__main__":
    main()
