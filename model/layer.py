import torch
from torch import nn
from hurst import compute_Hc, random_walk


def cal_hurst(feats):
    n = feats.size(0)
    # H, c, data = compute_Hc(series, kind='price', simplified=True)
    H = []
    for i in range(n):
        # feats是tensor张量，函数的参数是numpy矩阵
        # .detach().numpy() 将张量转化为numpy数组
        h, _, _ = compute_Hc(feats[i].detach().numpy(), kind='price')
        H.append(h)

    #print(H)
    #print(len(H))
    return H


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
    feats = torch.rand([200, 150])
    print(feats)
    # o, h, c = lstm.forward(feats, activations=[nn.Sigmoid(), nn.Tanh()])
    cal_hurst(feats)
    # print(c)
    # print(h)
    # print(o)
