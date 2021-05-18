"""
计算hurst参数
"""
import numpy as np


ts = np.random.rand(1, 30)
N = ts.shape[0]
d = ts.shape[1]

for i in range(10, d//2 + 1):
    """
    序列分解计算：
    1、子序列计算均值
    2、计算离差
    3、计算累积离差
    4、计算极差
    5、计算标准差
    6、计算RS值
    """
    # 将序列分解互不重叠的子序列
    subset_list = [ts[:, j: j+i] for j in range(0, d, i)]
    print(subset_list)
    if np.mod(d, i) > 0:
        # 去掉最后一个，不完整
        pass




