import numpy as np
import matplotlib.pyplot as plt
from minst import load_mnist
from Multilayer import MultiLayerNetExtend
from function import *
from layers import *
from active import *

def mynn():

    """
    normalize设置是否将输入图像正规化为0.0～1.0的值
    fatten设置是否展开输入图像（变成一维数组）
        如果将该参数设置为False，则输入图像为1 × 28 × 28的三维数组；若设置为True，则输入图像会保存为由784 个元素构成的一维数组。
    one_hot_label设置是否将标签保存为one-hot 表示。one-hot表示是仅正确解标签为1，其余皆为0的数组，
        当 one_hot_label 为 False时，只是像7、2这样简单保存正确解标签；
        当 one_hot_label 为 True时，标签则保存为one-hot表示。
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    # 返回形式：(训练图像 , 训练标签 )，( 测试图像，测试标签 )
    # network = Linear(input_size=784, hidden_size_list=[200,100,100], output_size=10)

    # 扩展神经网络的参数设定
    # 设定是否使用Dropuout，以及比例 ========================
    use_dropout = True  # 不使用Dropout的情况下为False
    dropout_ratio = 0.5
    # 设定使用的激活函数以及对应的权重值 ======================
    weight_init = 'relu'
    # 设定是否使BatchNormalization =======================
    use_bat = True


    network = MultiLayerNetExtend(input_size= 784,hidden_size_list=[200,64,64],output_size=10,
                                  activation =weight_init,weight_init_std = weight_init,
                                weight_decay_lambda = 0,use_dropout = use_dropout, dropout_ration = dropout_ratio,use_batchnorm = use_bat)

    iter_num = 20000  # 循环的次数
    train_size = x_train.shape[0]
    batch_size = 100  # 取出元素的个数
    # learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # optimizer = SGD(learning_rate=0.1)
    # optimizer = Momentum(learning_rate=0.1,momentum=0.9)
    # optimizer = AdaGrad(learning_rate = 0.1 )
    optimizer = Adam(learning_rate = 0.001 )

    # 平均每个epoch的重复次数
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iter_num):
        #  获取mini_batch
        batch_mask = np.random.choice(train_size, batch_size)  # 随机生成batch_size 个数（下标
        x_batch = x_train[batch_mask]  # 取出这些下标的元素
        t_batch = t_train[batch_mask]  # 取出这些下标的元素

        # 计算梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新梯度
        optimizer.update(network.params,grad)
        """
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]
        """

        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("训练集精确度：{:.3f},测试集精确度：{:.3f}".format(train_acc,test_acc))

    #  绘制loss图形
    print(train_loss_list)
    x = np.arange(len(train_loss_list))
    plt.plot(x,train_loss_list)
    plt.xlabel("x")
    plt.ylabel("loss")
    plt.ylim(0,3.00)
    plt.legend(loc='best')
    plt.show()

    # 绘制图形
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    mynn()

