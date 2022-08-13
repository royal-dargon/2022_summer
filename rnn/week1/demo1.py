import numpy as np
import rnn_utils


# 将开始向量化m个样本，x的维度是（nx，m），a的维度是（na，m）
def rnn_cell_forward(xt, a_prev, parameters):
    """""
    实现rnn单元的单步前向传播
    参数：
        xt：每个时间步进行输入的数据（nx，m）
        a_prev：前一个时间步的状态（na，m）
        parameters：字典，包含了单元内的一些参数信息
    """""

    # 从parameters中获取需要的参数,(na,nx) (na,na) (ny,na) (na,1) (ny,1)
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 使用上面的公式去计算下面一个的a的数值
    a_t = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # 计算本单元的输出
    y = rnn_utils.softmax(np.dot(Wya, a_t) + by)
    # 保存反向传播需要的数值
    cache = (a_t, a_prev, xt, parameters)
    return a_t, y, cache


# rnn是上面那些单元的重复连接，主要的步骤是初始化，随后循环所有的时间步
def rnn_forward(x, a0, parameters):
    """""
    实现一个循环神经网络的前向传播
    参数：
        x：输入的全部数据(nx,m,Tx)
        a0:初始化隐藏状态(na,m)
        parameters:
            Wax (n_a, n_x)
            Waa (n_a, n_a)
            Wya (n_y, n_a)
            ba (n_a, 1)
            by (n_y, 1)
    """""

    caches = []
    # 获取x，Wya的维度信息
    n_x, m, t_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # 使用0对a，y进行初始化
    a = np.zeros([n_a, m, t_x])
    y = np.zeros([n_y, m, t_x])

    # 初始化next
    a_next = a0
    # 开始对句子的长度进行遍历
    for t in range(t_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y, caches


# 接下来我们会去实现一个更加有效的lstm模型
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    实现一个LSTM单元的前向传播
    参数:
        xt 输入的数据，维度为(nx,m)
        a_prev 上一个时间步的隐藏状态,(n_a,m)
        c_prev 上一个时间步的记忆状态,(n_a,m)
        parameters: 包含了各种权值，遗忘门；更新门；输出门
    """

    # 从parameters中获取信息
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取xt，Wy的维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 连接a_prev与xt
    contact = np.zeros([n_a + n_x, m])
    contact[:n_a, :] = a_prev
    contact[n_a:, :] = xt

    # 遗忘门，公式1
    ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)
    it = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)
    cct = np.tanh(np.dot(Wc, contact) + bc)

    c_next = ft * c_prev + it * cct
    ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)
    a_next = ot * np.tanh(c_next)
    # 3.计算LSTM单元的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    # 保存包含了反向传播所需要的参数
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


