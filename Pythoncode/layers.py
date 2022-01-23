import numpy as np
from active import *
from function import *
from collections import OrderedDict


class Affine(object):
    """

    """

    def __init__(self, W, b):
        """
        :param W: 权重
        :param b: 偏置参数
        """
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss(object):
    """
    交叉损失熵层的实现
    由Softmax 与 loss 构成：
    """

    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据t ont-hot

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        Attention :将要传播的值除以批的大小（natch_size)
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


"""
以下实现几个常用的梯度函数
"""


class SGD(object):
    """
    SGD:随机梯度下降法
    Attention：函数非均向时，搜索的路径会低效
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]


class Momentum(object):
    """
    Momentum
    """

    def __init__(self, learning_rate, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum  # 对应公式中的$\alpha$
        self.v = None  # 速度

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] \
                          - self.learning_rate * grads[key]
            params[key] += self.v[key]


class AdaGrad(object):
    """
    AdaGrad：学习率衰减的思想
    可以通过RMSProp方法改善无止境学习后更新量变为0的问题
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(object):
    """
    Adam：融合 Momentum 与 AdaGrad
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


class Linear(object):
    """
    全连接层的实现:
    仅实现三层，防止过拟合
    """

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        """
        :param input_size: 输入层的参数:784
        :param hidden_size_list 隐藏层的神经元数量
        :param output_size: 输出层的参数:10
        :param weight_init_std: 权重
        """

        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size_list[0]),
                       'b1': np.zeros(hidden_size_list[0]),
                       'W2': weight_init_std * np.random.randn(hidden_size_list[0], output_size),
                       # 'b2': np.zeros(hidden_size_list[1]),
                       # 'W3': weight_init_std * np.random.randn(hidden_size_list[1], output_size),
                       # 'b3': np.zeros(hidden_size_list[2]),
                       # 'W4': weight_init_std * np.random.randn(hidden_size_list[2], output_size),
                       'b2': np.zeros(output_size)
                       }

        # 生成层
        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        """
        self.layers['Relu2'] = Relu()

        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        
        self.layers['Relu3'] = Relu()

        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        """
        self.lastLayer = SoftmaxWithLoss()

        # self.layers['Sigmoid'] = Sigmoid()
        # self.layers['tanh'] = tanh()
        # self.layers['Relu2'] = Relu()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        # 保存梯度的字典型常量
        grad = {'W1': numerical_gradient(loss_W, self.params['W1']),
                'b1': numerical_gradient(loss_W, self.params['b1']),
                'W2': numerical_gradient(loss_W, self.params['W2']),
                'b2': numerical_gradient(loss_W, self.params['b2'])
                }
        """
,
        'W3': numerical_gradient(loss_W, self.params['W3']),
        'b3': numerical_gradient(loss_W, self.params['b3'])
        'W4': numerical_gradient(loss_W, self.params['W4']),
        'b4': numerical_gradient(loss_W, self.params['b4'])
        """

        return grad

    # 误差反向传播法计算梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layers in layers:
            dout = layers.backward(dout)

        # 设定
        grads = {'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW, 'b2': self.layers['Affine2'].db,
                 # 'W3': self.layers['Affine3'].dW, 'b3': self.layers['Affine3'].db,
                 # 'W4': self.layers['Affine4'].dW, 'b4': self.layers['Affine4'].db
                 }
        return grads


"""
以下部分用于卷积神经网络的实现
"""


class Nesterov(object):
    """
    Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class RMSprop:
    """
    RMSprop
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Convolution(object):
    """
    卷积层的实现
    """

    def __init__(self, W, b, stride=1, pad=0):
        """

        :param W:
        :param b:
        :param stride:
        :param pad:
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    """
    池化层的实现
    """

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        """

        :param pool_h:
        :param pool_w:
        :param stride:
        :param pad:
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
