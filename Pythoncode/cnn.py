import pickle
import matplotlib.pyplot as plt
import numpy as np
from layers import *
from Multilayer import Dropout


class DeepConvNet:
    """
    网络结构如下所示
        conv - relu - conv- relu - pool - conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool - affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        # 初始化权重===========
        # 各层的神经元平均与前一层的几个神经元有连接（TODO:自动计算）
        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用ReLU的情况下推荐的初始值

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            # 组合为一个索引序列，同时列出数据和数据下标
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # ==========生成层===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        """
        :param x:输入数据
        :param train_flg: 训练标志
        :return:
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        :param x:输入数据
        :param t: 监督数据
        :return: ；loss值-->loss由最后一层交叉损失熵的前项传播计算
        """
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0  # 设定精确度初始值0.0

        # 批处理
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """
        梯度的计算
        :param x:
        :param t:
        :return:
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name='./pkl/params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="./pkl/params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]


class Trainer:
    """
    进行神经网络的训练的类
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'learning_rate': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        """
        :param network: 神经网络
        :param x_train: 训练集数
        :param t_train: 训练集的验证数据集
        :param x_test:  测试集的数据集
        :param t_test:  测试集的验证数据集
        :param epochs:  段
        :param mini_batch_size: 最小批
        :param optimizer: 梯度下降方式
        :param optimizer_param: 梯度下降函数需要的参数
        :param evaluate_sample_num_per_epoch:
        :param verbose:输出标志
        """
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer 梯度下降的方式
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):

        batch_mask = np.random.choice(self.train_size, self.batch_size)  # 获取mini_batch

        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)  # 计算损失
        self.train_loss_list.append(loss)  # 加入损失列表
        if self.verbose:
            print("train loss:" + str(loss))

        # 计算一批
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            # 计算梯度与精确度并加入列表
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(
                    "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                        test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
            print(self.train_loss_list)

            #  绘制loss图形
            x = np.arange(len(self.train_loss_list))
            plt.plot(x, self.train_loss_list)
            plt.xlabel("x")
            plt.ylabel("loss")
            plt.ylim(0, 2.5)

            plt.legend(loc='best')
            plt.show()

            # 绘制acc图形
            markers = {'train': 'o', 'test': 's'}
            x = np.arange(len(self.train_acc_list))
            plt.plot(x, self.train_acc_list, label='train acc')
            plt.plot(x, self.test_acc_list, label='test acc', linestyle='--')
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.ylim(0, 1.0)
            plt.legend(loc='lower right')

            plt.show()
