# 完成卷积神经网络对于 MNIST 数据集的构建与使用
from src.common.config import MNISTConfig
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as datasets
from src.common.utils import *
from torch.autograd import Variable
from torch import nn as nn
from torch import optim


class MNISTCNNModel(nn.Module):
    """
    搭建 CNN 循环神经网络来训练 MNIST 训练集.
    """

    def __init__(self, *args, **kwargs):
        """输入: [bs, 0, input_dim, input_dim] -> 输出: [bs, label_num]"""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),  # [bs, 16, 26, 26]
            nn.BatchNorm2d(16),  # 防止数值过大
            nn.ReLU()  # 激活函数
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(MNISTConfig.device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 8, 7),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ).to(MNISTConfig.device)
        self.linear1 = nn.Sequential(
            nn.Linear(16 * 16 * 8, 8 * 8 * 4),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(8 * 8 * 4, 4 * 4 * 2),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(4 * 4 * 2, 10)
        )

    def forward(self, inputs):
        temp = self.conv1(inputs)
        temp = self.conv2(temp)
        temp = self.conv3(temp)
        temp = self.conv4(temp)
        # flatten
        temp = temp.view((inputs.shape[0], -1))
        # 线性回归
        temp = self.linear1(temp)
        temp = self.linear2(temp)
        temp = self.linear3(temp)
        return temp

    def set_loss_function(self, loss_function):
        setattr(self, 'loss_function', loss_function)

    def set_optimizer(self, optimizer):
        setattr(self, 'optimizer', optimizer)

    def evaluate_forward(self, x_inputs, y_target):
        """前向传播一次, 并且返回一个 loss."""
        pred = self(x_inputs)
        loss = self.loss_function(pred, y_target)
        return pred, loss

    def train_one_batch(self, x_train, y_train):
        """完成一个 batch 的训练."""
        # 1. 首先前向传播, 计算一次
        y_pred, loss = self.evaluate_forward(x_train, y_train)
        # 2. 清空梯度
        self.optimizer.zero_grad()
        # 3. 反向传播计算梯度
        loss.backward()
        # 4. 优化器选择移动一个梯度的方位
        self.optimizer.step()
        # 5. 返回结果
        return y_pred, loss


class Main(object):

    @classmethod
    def train_and_save(cls):
        # 读入数据
        train_set = MNISTDataLoader.read_in_dataset(
            MNISTConfig.data_dir, 'train'
        )
        train_batch_vector_list = MNISTDataLoader.trans_data_set_2_batch_vector_list(
            train_set, MNISTConfig.batch_size, True
        )
        # 配置模型
        model = MNISTCNNModel()
        model.to(MNISTConfig.device)
        # model.to(MNISTConfig.device)
        loss_function = nn.CrossEntropyLoss()  # 注意这里每个 nn 中的对象, 都是一个 callable 的函数
        model.set_loss_function(loss_function)
        optimizer = optim.RMSprop(model.parameters(), lr=MNISTConfig.lr)
        model.set_optimizer(optimizer)
        # 开始训练
        model.train()
        for epoch in range(0, 2):
            for batch_index, batch_data_vector in enumerate(train_batch_vector_list):
                x_train = Variable(batch_data_vector[0])
                x_train = x_train.to(MNISTConfig.device)
                y_train = Variable(batch_data_vector[1])
                y_train = y_train.to(MNISTConfig.device)
                pred, loss = model.train_one_batch(x_train, y_train)
                print(
                    'epoch: {0}, batch: {1}, loss_val: {2:.3f}, acc: {3:.3f}'.format(
                        epoch, batch_index,
                        loss.item(),
                        calculate_accuracy(
                            mask_by_max_2d, judge_4_one_hot_multi_class_by_equal_2d,
                            pred, dim_1_tensor_2_one_hot(y_train, 10)
                        )
                    )
                )
        # 训练结束存储模型
        save_model(model, MNISTConfig.join_path('out', 'mnist/cnn.pkl'))

    @classmethod
    def load_and_test(cls):
        """完成对于模型的加载, 并且进行评价预测."""
        # 读取测试集
        dataset = MNISTDataLoader.read_in_dataset(MNISTConfig.data_dir, 'test')
        batch_vec_list = MNISTDataLoader.trans_data_set_2_batch_vector_list(
            dataset, MNISTConfig.batch_size, False
        )
        # 加载模型
        model = MNISTCNNModel()
        model = load_model(model, MNISTConfig.join_path('out', 'mnist/cnn.pkl'))
        loss_function = nn.CrossEntropyLoss()
        model.set_loss_function(loss_function)
        model.to(MNISTConfig.device)
        # 开始测试
        model.eval()
        # for batch_idx, (batch_data, batch_label) in enumerate(batch_vec_list):
        #     x_test = batch_data.to(MNISTConfig.device)
        #     y_test = batch_label.to(MNISTConfig.device)
        #     pred, loss = model.evaluate_forward(x_test, y_test)
        #     print(
        #         'batch: {0}, eval_loss: {1:.3f}, eval_acc:{2:.3f}'.format(
        #             batch_idx, loss.item(), calculate_accuracy(
        #                 mask_by_max_2d, judge_4_one_hot_multi_class_by_equal_2d, y_pred=pred,
        #                 y_target=dim_1_tensor_2_one_hot(y_test, 10)
        #             )
        #         )
        #     )
        temp_vec = next(iter(batch_vec_list))
        x_test = temp_vec[0][0].reshape((1, 1, 28, 28)).to(MNISTConfig.device)
        picture = x_test[0][0]
        y_pred = model(x_test)
        print(y_pred)
        show_one_2d_image(
            picture.to(torch.device('cpu')), str(y_pred.argmax().item())
        )


class MNISTDataLoader(object):
    @classmethod
    def read_in_dataset(cls, data_dir: str, dataset_type: str):
        """读入数据集."""
        is_train = True
        if dataset_type != 'train' and dataset_type != 'test':
            raise KeyError
        else:
            is_train = (dataset_type == 'train')
        transforms = trans.Compose([trans.ToTensor()])
        mnist_data_set = datasets.MNIST(
            data_dir, is_train, transforms
        )
        return mnist_data_set

    @classmethod
    def trans_data_set_2_batch_vector_list(cls, data_set, batch_size, shuffle):
        batch_vector_list = DataLoader(
            data_set, batch_size, shuffle
        )
        return batch_vector_list


if __name__ == '__main__':
    # Main.train_and_save()
    Main.load_and_test()
