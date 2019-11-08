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





## :star: 3. Pytorch 使用的注意点

1. 加载数据的思路

    1. 首先读入一个 dataset, 并完成一些初始化的工作
    2. 然后构造一个可以迭代的对象

    3. 注意导入的包

        ```python
        from torch.utils.data import DataLoader
        import torchvision.transforms as trans
        import torchvision.datasets as datasets
        ```

2. 模型的构建以及训练部分

    1. 模型的构建, 推荐使用 `nn.Sequential(..., ..., ...,)`

        ```python
        self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16, 3),  # [bs, 16, 26, 26]
                    nn.BatchNorm2d(16),  # 防止数值过大
                    nn.ReLU()  # 激活函数
                )
        ```

    2. 模型前向传播, 计算一次

        ```python
            def evaluate_forward(self, x_inputs, y_target):
                """前向传播一次, 并且返回一个 loss."""
                pred = self(x_inputs)
                loss = self.loss_function(pred, y_target)
                return pred, loss
        ```

    3. 训练一个 batch

        ```python
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
        ```

3. **<font color='red'>pytorch 的模块理解</font>**

    1. 使用了 $nn$ 模块, 那么其中所有包含的网络结构层, 以及损失函数, 都是一个 callable 的对象, 所以在往模型中传入损失函数的时候, 要注意传入一个对象

        ```python
        # 配置模型
                model = MNISTCNNModel()
                # model.to(MNISTConfig.device)
                loss_function = nn.CrossEntropyLoss()  # 注意这里每个 nn 中的对象, 都是一个 callable 的函数
                model.set_loss_function(loss_function)
                optimizer = optim.RMSprop(model.parameters(), lr=MNISTConfig.lr)
                model.set_optimizer(optimizer)
        ```

    2. 对于 $loss$ 来说, 是保留了之前的计算图的信息的
    3. 对于 $optimizer$ 来说, 他保留的则是这个模型的梯度以及权重, 偏置的参数信息

4. **注意模型的训练, 保存; 读取, 评价过程**
    * 保存其实保存的是权重字典
    * 读取也是读入权重字典
    * 注意: 
        * train: 在训练前加上 `model.train()`
        * eval: 在测试前加上 `model.eval()`

5. 注意使用 `cuda` 加速, 要对 model 以及输入的张量都 `.to(device)`
   
    * **注意: 这个是有返回值的, 需要被赋值!!!**





## 4. 数据的初步处理

1. 数据增强
2. 数据预处理: 归一化是指成为 0 均值 (高斯 normal)







## 5. 深度学习基础模型

### (1) CNN-Net

1. 其中的卷积操作是: 覆盖后进行对象元素相乘, 然后相加

2. 计算公式: $ o = (i + 2p - k + 1)/s , i = input\_size, p = padding\_size, k = kernel\_size$

3. 注意: 

    1. 卷积之后的一层可以是池化, 也可以是激活函数

        ```python
        		self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16, 3),  # [bs, 16, 26, 26]
                    nn.BatchNorm2d(16),  # 防止数值过大
                    nn.ReLU()  # 激活函数
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(16, 32, 3),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(32, 16, 3),
                    nn.BatchNorm2d(16),
                    nn.ReLU()
                )
                self.conv4 = nn.Sequential(
                    nn.Conv2d(16, 8, 7),
                    nn.BatchNorm2d(8),
                    nn.ReLU()
                )
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
        ```

    2. 注意线性回归的最后一层, 往往是不加入激活函数的

        ```python
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
        ```





### (2) nn.CrossEntropyLoss()

1. 交叉信息商函数的定义

    *  ![img](F:\Research\ComputerVision\pictures\cross_entropy.png)

    * 假设x是正确的概率分布，而y是我们预测出来的概率分布，这个公式算出来的结果，表示y与正确答案x之间的错误程度（即：y错得有多离谱），结果值越小，表示y越准确，与x越接近。

    * 比如: 

        x的概率分布为：{1/4 ，1/4，1/4，1/4}，现在我们通过机器学习，预测出来二组值：

        y1的概率分布为 {1/4 , 1/2 , 1/8 , 1/8}

        y2的概率分布为 {1/4 , 1/4 , 1/8 , 3/8}

    * 从直觉上看，y2分布中，前2项都100%预测对了，而y1只有第1项100%对，所以y2感觉更准确，看看公式算下来，是不是符合直觉: 

    ![img](F:\Research\ComputerVision\pictures\cross_entropy_1.png)

     

     	![img](F:\Research\ComputerVision\pictures\cross_entropy_2.png)

    * **对比可以发现, 交叉信息熵可以用来较为真实的反映两组预测之间的概率分布偏差程度**

2. **注意, pytorch 中的交叉信息商函数, 输入的形式 $input: [bs, class\_num], target: [bs,]$**

