import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 生成数据集
def create_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
X, Y = create_data(true_w, true_b, 1000)

plt.scatter(X[:, (1)].detach().numpy(), Y.detach().numpy(), 1)
plt.show()
