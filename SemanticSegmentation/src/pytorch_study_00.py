# 用于学习 pytorch 下的深度学习的学习
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Utils(object):
    """工具类."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256

    @classmethod
    def print_tensor(cls, tensor):
        print(tensor)
        print(tensor.shape)

    @classmethod
    def print_with_type(cls, what):
        """打印内容, 并且显示类型."""
        print(what)
        print(type(what))

    @classmethod
    def unnormalize(cls, img_tensor: torch.tensor, mean, std):
        """将一个图片还原回没有 normalize 之前的样子."""
        ans_tensor = torch.zeros_like(img_tensor)
        for channel_index, channel_tensor in enumerate(img_tensor):
            # 获得到每一个 channel 上的张量
            ans_tensor[channel_index] = (
                    channel_tensor * std[channel_index] +
                    mean[channel_index]
            )
        return ans_tensor

    @classmethod
    def print_img_tensor(cls, img_tensor, label_title):
        """打印一张图片张量.
        注意: 由于 plt 默认认为图片是 H * W * C
        而数据集处理的图片是 C * H * W
        """
        plt.figure()
        plt.title(label_title)
        plt.imshow(
            img_tensor[0],
            cmap='gray',  # 绘制黑白色
            interpolation='none'  # 不适用插值算法
        )
        plt.show()


class MNISTDataLoader(object):
    """MNIST 数据集的加载对象."""
    mean = (0.1307,)
    std = (0.381,)

    def get_data_tensors(self, data_set, need_shuffle=True):
        """将原始数据集转换成一个可以迭代的对象."""
        return torch.utils.data.DataLoader(
            dataset=data_set, batch_size=Utils.batch_size,
            shuffle=need_shuffle
        )

    def load_raw_data_set(
            self, data_set_dir, need_download=True, is_train_set=True
    ):
        """在这个方法中加载未处理的数据集."""
        # 数据集的图像预处理操作
        transforms_list = torchvision.transforms.Compose(
            [
                # 转化一张图片变成 Channel * Height * Width
                torchvision.transforms.ToTensor(),
                # 将输入的 0 ~ 1 上的图片像素 (what / 255) 变成一个在一个区间范围内的分布数值
                # 注意 MNIST 是单通道的, 所以只需要一个维度的元组即可
                # 具体的数值是一个超参数, 经过训练可以确定优化的.
                torchvision.transforms.Normalize(
                    MNISTDataLoader.mean,
                    MNISTDataLoader.std
                )
            ]
        )
        # 下载训练集, 返回一个 dataset 对象
        data_set = torchvision.datasets.MNIST(
            data_set_dir, train=is_train_set,  # 下载为训练集
            download=need_download, transform=transforms_list
        )
        return data_set


class MNISTLinearModel(nn.Module):
    """搭建一个特定网络结构的神经网络模型."""

    def __init__(self):
        super(MNISTLinearModel, self).__init__()
        # 配置网络层
        self.h1 = nn.Linear(28 * 28, 256)
        self.h2 = nn.Linear(256, 64)
        self.h3 = nn.Linear(64, 10)
        # self.loss_function = _loss_function

    def forward(self, input_x):
        # 前向传播
        h1_y = F.relu(self.h1(input_x))
        h2_y = F.relu(self.h2(h1_y))
        output_y = self.h3(h2_y)
        return output_y

    def train_a_batch(self, x_train, y_train):
        """
        训练一个 batch 的数据.
        0. 注意向量维度, 对于这个模型, 使用的时 [batch_size, features]
        1. 前向传播一次 (可以直接用对象名, 使用 callable 方法, 也可以 forward 一次)
        2. 计算损失值
        3. 清空当前上一次的梯度值
        4. 反向传播计算梯度
        5. 使用优化器, 寻找一个下降的道路, 使得整个权重与偏置被更新
        """
        # 1. 前向传播, 获取当前计算值
        y_predict = self(x_train)
        # 2. 计算损失值
        loss = self.loss_function(y_predict, y_train)
        loss_val = loss.item()
        # 3. 清空上一次计算的梯度值 (Pytorch 中的 Variable 会保留梯度值)
        # 因为是绑定的, 所以清空很重要, 清空后计算, 不然会使用上一次的信息
        self.optimizer.zero_grad()
        # 4. 反向传播, 计算梯度
        loss.backward()
        # 5. 优化器使用当前计算的梯度信息向前走一步
        self.optimizer.step()
        return y_predict, loss_val

    @classmethod
    def trans_batch_data_tensor_2_one_tensor(
            cls, batch_data_tensor: torch.Tensor
    ):
        """将输入个一组图片, 拉成一组一条的 tensor, 然后合并成一个矩阵."""
        return batch_data_tensor.view((batch_data_tensor.shape[0], -1))

    @classmethod
    def trans_batch_labels_2_one_hot_tensor(
            cls, batch_label_tensor, label_num
    ):
        """将一个 tensor 转成一个 [batch_size, label_num]."""
        ans_tensor = torch.zeros((batch_label_tensor.shape[0], label_num))
        for index, label in enumerate(batch_label_tensor):
            ans_tensor[index][label] = 1.0
        return ans_tensor

    def save_model_to(self, to_path):
        """将一个模型, 保存到指定的位置."""
        current_state_dict = self.state_dict()
        torch.save(current_state_dict, to_path)


def test():
    """
    1. 读入模型.
    2. 运行测试集.
    :return:
    """
    # 首先读入模型
    model = MNISTLinearModel()
    model.load_state_dict(torch.load('../out/mnist_01.pkl'))
    setattr(model, 'loss_function', F.mse_loss)
    setattr(
        model, 'optimizer',
        optim.SGD(model.parameters(), lr=0.005)
    )
    model.eval()
    # 读入测试集
    mnsit_data_loader = MNISTDataLoader()
    test_data_set = mnsit_data_loader.load_raw_data_set(
        data_set_dir='../data/',
        need_download=False, is_train_set=False
    )
    mnsit_data_set = mnsit_data_loader.get_data_tensors(
        test_data_set, False
    )
    # 进行测试
    for test_batch_idx, test_batch_list in enumerate(mnsit_data_set):
        test_batch = test_batch_list[0]
        test_batch_label = test_batch_list[1]
        x_test = MNISTLinearModel.trans_batch_data_tensor_2_one_tensor(
            test_batch
        )
        y_test = MNISTLinearModel.trans_batch_labels_2_one_hot_tensor(
            test_batch_label, 10
        )
        y_predict, y_loss = model.train_a_batch(
            x_test, y_test
        )
        print(y_loss)
        where = str(y_predict[0].argmax().item())
        true = int(test_batch_label[0].item())
        print(where)
        print(true)
        Utils.print_img_tensor(
            test_batch[0], 'pred: {}; true: {}'.format(
                where, true
            )
        )
        input()


def main():
    # 训练集的训练
    mnsit_data_loader = MNISTDataLoader()
    train_data_set = mnsit_data_loader.load_raw_data_set(
        data_set_dir='../data/',
        need_download=False, is_train_set=True
    )
    epoch_data_list = mnsit_data_loader.get_data_tensors(
        data_set=train_data_set, need_shuffle=True
    )
    # 首先确定损失函数
    loss_function = F.mse_loss
    # 确定模型
    model = MNISTLinearModel()
    model = model.to(Utils.device)
    setattr(model, 'loss_function', loss_function)
    # 绑定并且添加优化器
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    setattr(model, 'optimizer', optimizer)
    # 开始训练 epoch
    for epoch in range(3):
        # print((' %d ' % epoch).center(31, '*'))
        for batch_index, batch_data_with_labels_list in enumerate(
                epoch_data_list
        ):
            # 获取一个 batch 的数据, 每个都是一个单通道的图片
            batch_data_tensor = batch_data_with_labels_list[0]
            x_train = MNISTLinearModel.trans_batch_data_tensor_2_one_tensor(
                batch_data_tensor
            )
            x_train = x_train.to(Utils.device)
            # 一个 tensor: [batch_size]
            batch_label_tensor = batch_data_with_labels_list[1]
            y_train = MNISTLinearModel.trans_batch_labels_2_one_hot_tensor(
                batch_label_tensor, 10
            )
            y_train = y_train.to(Utils.device)
            # 训练一个 batch
            loss = model.train_a_batch(x_train, y_train)[1]
            print(
                'epoch: %d, batch: %d, loss: %.4f' % (
                    epoch, batch_index, loss
                )
            )
    # 保存模型
    model.save_model_to('../out/mnist_01.pkl')
    # model.state_dict()
    # 开始测试


if __name__ == "__main__":
    # main()
    test()
