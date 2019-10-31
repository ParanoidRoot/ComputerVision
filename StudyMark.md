# Study Mark



## 1. Pytorch 的数据集加载与使用方式

0. <font color='red' size=5>Pytorch 中采用的矩阵结构</font>
    
    1. 对于输入 $x_{train}:~case * features$
    2. 对于输出 $y_{train}:~case*labels$
3. 注意图像的输入 $img:~C*H*W$, 代表了 Channel * Height * Width
    
1. 首先要加载原生的数据集

    ```python
    def load_raw_data_set(
            self, data_set_dir, need_download=True, is_train_set=True
        ):
            '''在这个方法中加载未处理的数据集.'''
            # 数据集的图像预处理操作
            transforms_list = torchvision.transforms.Compose(
                [
                    # 转化一张图片变成 Channel * Height * Width
                    torchvision.transforms.ToTensor(),
                    # 将输入的 0 ~ 1 上的图片像素 (what / 255) 变成一个在一个区间范围内的分布数值
                    # 注意 MNIST 是单通道的, 所以只需要一个维度的元组即可
                    # 具体的数值是一个超参数, 经过训练可以确定优化的.
                    torchvision.transforms.Normalize((0.1307,), (0.381,))
                ]
            )
            # 下载训练集, 返回一个 dataset 对象
            data_set = torchvision.datasets.MNIST(
                data_set_dir, train=is_train_set,  # 下载为训练集
                download=need_download, transform=transforms_list
            )
            return data_set
    ```

2. 然后使用 `torch.utils.data.DataLoader` 将原始数据集, 转化成 pytorch 中的 tensor 对象

    ```python
    def get_data_loader(self, data_set, need_shuffle=True):
            '''将原始数据集转换成一个可以迭代的对象.'''
            return torch.utils.data.DataLoader(
                dataset=data_set, batch_size=Utils.batch_size,
                shuffle=need_shuffle
            )
    ```

3. **<font color='red'>迭代获取数据</font>**

    ```python
    	epoch_data_list = mnsit_data_loader.get_data_tensors(
            data_set=data_set, need_shuffle=True
        )
        # Utils.print_with_type(data_loader)
        for index, batch_data_list in enumerate(epoch_data_list):
            # 每个迭代是一个列表
            # 第一维是一个 batch 的数据的tensor, 每个元素是一个图片, shape: [batch_size, C, H, W]
            batch_train_data_tensor = batch_data_list[0]
            # 第二维是一个 batch 的标签的tensor, 每一个维度的值, shape: [batch_size,]
            labels = batch_data_list[1]
    ```





## 2. Pytorch 的训练过程

1. 第一步获取数据集
    
    * 在目录 `1. Pytorch 的数据集加载和使用方式` 中已经阐明了
2. 构建模型
    1. 确定每层网络的结构
    2. 确定 `forward` 方法中的每个 batch 的输入相应所做的运算, 包括激活函数等等, 返回一个结果向量, 要与 $y_{train}$ 的 $features$ 数目一样

3. 确定损失函数 $loss\_fuction$

4. 确定需要用到的 $optimizer$, 并且绑定到模型对象上

    1. ```python
        # 绑定并且添加优化器
            optimizer = optim.SGD(model.parameters(), lr=0.005)
            setattr(model, 'optimizer', optimizer)
        ```

    2. 

5. 训练一个 batch 的过程

    1. $forward$ 向前计算一次, 获取输出的 $y_{predict}$
    2. 计算出当前的损失值 $loss = self.loss\_function(y_{predict}, y_{train})$
        * 注意, 这个 $loss$ 中会保存动态的计算流图
        * **<font color='red'>向前计算梯度时, 使用的是 $loss$ </font>**
        * 获取损失值: `loss_val = loss.item()`

    3. 清空上一次计算的梯度剩余值
        
    * `self.optimizer.zero_grad()`
        
    4. 向前传播, 计算梯度
        
        * `loss.backward()`
    5. 使用优化器, 更新当前的权重, 向着利用损失函数的方向移动一个 $lr$
        
    * `self.optimizer.step()`
        
    6. 综合全部

        ```python
        def train_a_batch(self, x_train, y_train):
                '''
                训练一个 batch 的数据.
                0. 注意向量维度, 对于这个模型, 使用的时 [batch_size, features]
                1. 前向传播一次 (可以直接用对象名, 使用 callable 方法, 也可以 forward 一次)
                2. 计算损失值
                3. 清空当前上一次的梯度值
                4. 反向传播计算梯度
                5. 使用优化器, 寻找一个下降的道路, 使得整个权重与偏置被更新
                '''
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
                return loss_val
        ```

6. 每个 $epoch$

    ```python
     # 首先确定损失函数
        loss_function = F.mse_loss
        # 确定模型
        model = MNISTLinearModel(loss_function)
        model = model.to(Utils.device)
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
                loss = model.train_a_batch(x_train, y_train)
                print(
                    'epoch: %d, batch: %d, loss: %.4f' % (
                        epoch, batch_index, loss
                    )
                )
    ```





## 3. CNN & FCN

### (1) CNN

1. 其中的卷积操作是: 覆盖后进行对象元素相乘, 然后相加
2. 计算公式: $ o = \lfloor i + 2p - k\rfloor/s +1, i = input\_size, p = padding\_size, k = kernel\_size$
3. 