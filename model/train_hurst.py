import numpy as np
import torch
from torch import nn
from hurst import compute_Hc, random_walk
from make_data import read_data
import random
import pandas as pd


def make_data():
    datax, datay = read_data('data.xlsx')
    #[]变为（[]） 将序列型对象转化为数组
    datax = np.array(datax)
    datay = np.array(datay)
    random_index = random.sample(list(range(datax.shape[0])), datax.shape[0])
    #shape[0]数组行数，生成指定列表            从0到行数的列表，list将元组转化为列表
    train_size = int(datax.shape[0] * 0.6)#转化为整型
    train_index = random_index[:train_size] #索引范围
    test_index = list(set(random_index).difference(set(train_index)))
    train_feature = datax[train_index]
    train_label = datay[train_index]
    test_feature = datax[test_index]
    test_label = datay[test_index]
    return train_feature, train_label, test_feature, test_label

def cal_hurst(feats):
    n = feats.size(0)
    # H, c, data = compute_Hc(series, kind='price', simplified=True)
    H = []
    for i in range(n):
        # feats是tensor张量，函数的参数是numpy矩阵
        # .detach().numpy() 将张量转化为numpy数组
        h, _, _ = compute_Hc(feats[i].detach().numpy(), kind='price')
        H.append(h)

    print(H)
    print(len(H))
    return H


# def cal_hurst(ts):
#     ts = list(ts)#将时间序列转化为列表
#     print(ts)
#     N = len(ts)#N为时间序列长度
#     if N < 20:
#         raise ValueError("Time series is too short! input series ought to have at least 20 samples!")
#
#     max_k = int(np.floor(N / 2))
#     R_S_dict = []#创建一个空列表
#     for k in range(10, max_k + 1):
#         R, S = 0, 0
#         # split ts into subsets将时间序列分成子集
#         subset_list = [ts[i:i + k] for i in range(0, N, k)]
#         if np.mod(N, k) > 0:
#             subset_list.pop()
#             # tail = subset_list.pop()
#             # subset_list[-1].extend(tail)
#         # calc mean of every subset计算每个子集的均值
#         mean_list = [np.mean(x) for x in subset_list]
#         for i in range(len(subset_list)):
#             cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()#离差序列
#             R += max(cumsum_list) - min(cumsum_list)
#             S += np.std(subset_list[i])#求标准差
#         R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})
#
#     log_R_S = []
#     log_n = []
#     print(R_S_dict)
#     for i in range(len(R_S_dict)):
#         R_S = (R_S_dict[i]["R"] + np.spacing(1)) / (R_S_dict[i]["S"] + np.spacing(1))
#         log_R_S.append(np.log(R_S))
#         log_n.append(np.log(R_S_dict[i]["n"]))
#
#     Hurst_exponent = np.polyfit(log_n, log_R_S, 1)[0]#采用最小二次拟合
#     return Hurst_exponent


class LSTM(nn.Module):
    """
    参数输入：h，c作为循环变量
    加一个判断：
    if c = None, 则进行初始化
    else 采用上一层的输出
    """

    def __init__(self, **kwargs):#定义类的初始化函数
        #**kwargs是将一个可变的关键字参数的字典传给函数实参,为什么要有kwargs
        super(LSTM, self).__init__()

        self.dim_in = kwargs['dim_in']#输入的维度，dim_in是变量，kwargs是形参
        self.dim_out = kwargs['dim_out']#输出的维度
        self.f = nn.Linear(self.dim_in[0], self.dim_out[0])
        #nn.Linear用于设置全连接层，输入输出是二位向量，形状为（batch_size,size)
        #nn.Linear(in_features,out_features)
        self.i = nn.Linear(self.dim_in[0], self.dim_out[0])
        self.ct = nn.Linear(self.dim_in[0], self.dim_out[0], bias=True)
        self.o = nn.Linear(self.dim_in[0], self.dim_out[0], bias=True)
        self.h = None
        self.c = None

    def get_state(self):
        """
        判断h、c是否存在
        :return:
        """
        shape_h = torch.empty(self.dim_in[0], self.dim_in[0])
        shape_c = torch.empty(8, 2)
        if self.h is None:
            self.h = torch.nn.init.xavier_uniform_(shape_h)
            #一个服从均匀分布的Glorot初始化器
        if self.c is None:
            self.c = torch.nn.init.xavier_uniform_(shape_c)

        return self.h, self.c

    def forward(self, feats, activations):#定义前向
        self.h, self.c = self.get_state()
        feats = torch.cat((feats, self.h), dim=0)#在给定维度上对输入的张量序列seq 进行连接操作
        #feats是x?
        # 添加hurst参数，维度和self.f输出维度一致
        f = activations[0](self.f(feats) + cal_hurst(feats))
        # print(f.size())
        # print(self.c.size())
        # 加hurst
        i = activations[0](self.i(feats))
        ct = activations[1](self.ct(feats))
        self.c = f * self.c + i * ct
        o = activations[0](self.o(feats))
        self.h = o * activations[1](self.c)
        return o, self.h, self.c



if __name__ == '__main__':
    dims_in = [5, 2]
    dims_out = [2, 2]
    lstm = LSTM(dim_in=dims_in, dim_out=dims_out)
    #feats = torch.rand([200, 150])
    path = 'data.xlsx'
    data = pd.read_excel(path, sheet_name=0)  # data为整张excel表格
    data_list = data.values.tolist()  # 获取数据，去除索引号,变成二维数组[[ ]]
    feats = torch.tensor(data_list)
    # o, h, c = lstm.forward(feats, activations=[nn.Sigmoid(), nn.Tanh()])
    cal_hurst(feats)
    # print(c)
    # print(h)
    # print(o)

    def train_lstm(model, x_train, x_test, y_train, y_test):
        # if os.path.exists('./weights/lstm-ss22.h5'):
        #     print('加载预训练模型')
        #     model = load_model('./weights/lstm-ss3.h5')
        # else:
        #     print('第一次训练')
        #     model = model
        print('start training!')

        # batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(x_test, y_test, batch_size=32)))
        model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_test, y_test))
        model.summary()
        print('start test')
        scores = model.evaluate(x_test, y_test, batch_size=32)
        print(scores)


    if __name__ == '__main__':
        x_train, y_train, x_test, y_test = make_data()
        model = LSTM()
        train_lstm(model, x_train, x_test, y_train, y_test)

        print('save model')
        model.save('./weights/myself_model227.h5')