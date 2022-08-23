import pandas as pd
import matplotlib.pyplot as plt


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
    plt.legend()
    plt.show()


# fit my data to the func
def fit_data():
    pass


def main():
    all_eth, all_c4 = get_data()
    print_plot(all_eth, all_c4)


if __name__ == "__main__":
    main()

