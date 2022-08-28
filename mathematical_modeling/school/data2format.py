import pandas as pd
import numpy as np


# this is the func of read the data
def get_data(file_name):
    data = pd.read_csv(file_name)
    print(data.shape)
    data_list = []
    for i in range(data.shape[1]):
        data_list.append(data.iloc[:, i])
    return np.array(data_list).astype(float)


# write to the txt
def write_csv(data):
    csv_dic = {}
    white = []
    for i in range(data.shape[1]):
        white.append("gene" + str(i + 1))
    csv_dic[" "] = white
    for i in range(data.shape[0] - 1000):
        name = "cell" + str(i+1)
        csv_dic[name] = list(data[i])
    df = pd.DataFrame(csv_dic)
    df.to_csv("test.csv", index=False, sep=',')


def main():
    data = get_data("data/data1.csv")
    write_csv(data)


if __name__ == "__main__":
    main()
