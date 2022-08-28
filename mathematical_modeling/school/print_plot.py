import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

problem_1 = [0.5437, 0.6069, 0.6123, 0.5916, 0.5700]
problem_2 = [0.6499, 0.7378, 0.7049, 0.6718, 0.6501]

problem_1_1 = [23.7907, 17.9663, 9.3908, 8.5544, 13.4063]
problem_1_2 = [7.2169, 6.6556, 4.7296, 4.3637, 4.0715]
x = [2, 5, 10, 15, 20]

plt.plot(x, problem_1, label="Observed_data1准确率")
# plt.scatter(x, problem_1, label="文件一准确率")
plt.plot(x, problem_2, label="Observed_data2准确率")
# plt.scatter(x, problem_2, label="文件二准确率")

plt.xlabel("聚类类别数")
plt.ylabel("dropout判断准确率")

plt.legend()
plt.show()

plt.plot(x, problem_1_1, label="Observed_data1评估效果")
# plt.scatter(x, problem_1, label="文件一准确率")
plt.plot(x, problem_1_2, label="Observed_data2评估效果")
# plt.scatter(x, problem_2, label="文件二准确率")

plt.xlabel("聚类类别数")
plt.ylabel("生成数与实际误差")

plt.legend()
plt.show()

