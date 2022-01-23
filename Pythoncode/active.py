import numpy as np

"""
本文档存放几个常见的激活函数,并实现计算图前向传播与反向传播
Relu(),Sigmoid(),tanh()
"""


class Relu(object):
    """Relu 激活函数
    """

    def __init__(self):
        # 实例变量mask，由True/False构成的NumPy数组
        # 正向传播是输入的x小于等于0保存为True
        self.mask = None

    def forward(self, _input):
        """
        Relu激活正向过程
        :param _input:
        :return:
        """
        mask = np.array(_input <= 0)  # _input <= 0 正向传播传向下一层0
        self.mask = mask
        out = _input.copy()  # out 赋值为 _input
        out[self.mask] = 0  # self.mask 为True 的置为0
        return out

    def backward(self, preGard):
        """
        Relu激活反向过程
        :param preGard:
        :return:
        """
        preGard[self.mask] = 0  # numpy数组,并将self.mask的置为0
        dx = preGard
        return dx


class Sigmoid(object):
    """Sigmoid 激活函数
    """

    def __init__(self):
        self.out = None

    def forward(self, _input):
        """
        Sigmoid激活正向过程
        :param _input:
        :return:
        """
        out = 1 / (1 + np.exp(- _input))
        self.out = out
        return out

    def backward(self, preGrad):
        """
        Sigmoid激活反向过程
        :param preGrad:
        :return:
        """
        dx = preGrad * (1.0 - self.out) * self.out
        return dx


class tanh(object):
    """"tanh激活函数
    """

    def __init__(self):
        self.out = None

    def forward(self, _input):
        """
        tanh激活前向过程
        :param _input:
        :return:
        """
        self.out = np.tanh(_input)
        return np.tanh(_input)

    def backward(self, preGrad):
        """
        tanh激活反向过程
        :param preGrad:
        :return:
        """
        return preGrad([1 - np.square(self.out)])
