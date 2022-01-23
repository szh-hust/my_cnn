from sklearn import datasets
import matplotlib.pyplot as plt
from layers import *
from active import *
from Multilayer import *


def mynn_sklearn():
    digits = datasets.load_digits()

    # 分割训练集与测试集
    x_train = digits.data[:1000]
    x_test = digits.data[1000:1796]
    t_train = digits.target[:1000]
    t_test = digits.target[1000:1796]

    network = Linear(input_size=64, hidden_size_list=[32], output_size=10)

    # 扩展神经网络的参数设定
    # 设定是否使用Dropuout，以及比例 ========================
    # use_dropout = True  # 不使用Dropout的情况下为False
    # dropout_ratio = 0.1
    # 设定使用的激活函数以及对应的权重值 ======================
    # weight_init = 'relu'
    # 设定是否使BatchNormalization =======================
    # use_bat = True

    """
    network = MultiLayerNetExtend(input_size=64, hidden_size_list=[32, 16, 10], output_size=10,/
                                  activation=weight_init, weight_init_std=weight_init,/
                                  weight_decay_lambda=0, use_dropout=use_dropout, dropout_ration=dropout_ratio,/
                                  use_batchnorm=use_bat)
                                  """

    iter_num = 300  # 循环的次数
    train_size = x_train.shape[0]
    batch_size = 50  # 取出元素的个数

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # optimizer = SGD(learning_rate=0.3)
    # optimizer = Momentum(learning_rate=0.1,momentum=0.9)
    # optimizer = AdaGrad(learning_rate=0.01)
    optimizer = Adam(learning_rate=0.01)

    for i in range(iter_num):
        #  获取mini_batch
        batch_mask = np.random.choice(train_size, batch_size)  # 随机生成batch_size 个数（下标
        x_batch = x_train[batch_mask]  # 取出这些下标的元素
        t_batch = t_train[batch_mask]  # 取出这些下标的元素

        # 计算梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新梯度
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("训练集精确度：{:.3f},测试集精确度：{:.3f}".format(train_acc, test_acc))

    print(train_loss_list)
    #  绘制loss图形
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list)
    plt.xlabel("x")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)

    plt.legend(loc='best')
    plt.show()

    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    mynn_sklearn()
