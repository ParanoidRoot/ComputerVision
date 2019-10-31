"""
在这个文件中完成对通用属性的配置.
"""
from matplotlib import pyplot as plt
import torch


def print_with_type(what):
    print('type is {}'.format(type(what)))
    print(what)


def print_shape(tensor):
    print(tensor.shape)


def show_one_2d_image(img_tensor, img_title=None):
    plt.figure()
    img_title = (img_title if img_title is not None else '')
    print(img_title)
    plt.title(img_title)
    plt.imshow(img_tensor, cmap='gray', interpolation='none')
    plt.show()
    plt.close()


def save_model(model, to_path):
    """将一个 torch 的模型输出保存到指定路径."""
    temp = model.state_dict()
    import os
    to_dir = os.path.dirname(to_path)
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    torch.save(temp, to_path)


def load_model(model, from_path):
    """将模块装在进来."""
    temp = torch.load(from_path)
    model.load_state_dict(temp)
    return model


def dim_1_tensor_2_one_hot(dim_1_tensor, label_num):
    """将一个一维的 tensor 转换成一个指定 label 个数的 one hot 编码."""
    ans = torch.zeros((dim_1_tensor.shape[0], label_num))
    for index, value in enumerate(dim_1_tensor):
        ans[index][value] = 1.0
    return ans


def calculate_accuracy(mask_function, judge_function, y_pred, y_target):
    """注意 y_pred 与 y_target 的 shape 是一样的, 通过某个`掩盖函数`, 应该要
    可以输出准确的 y_target 值, 最后就可以计算准确度.
    """
    y_pred = mask_function(y_pred)
    return judge_function(y_pred, y_target)


def mask_by_max_2d(y_pred):
    """将一个 batch size 中的每一个行的最大值挑选出来."""
    ans = torch.zeros_like(y_pred)
    for case, pred_case in enumerate(y_pred):
        i = pred_case.argmax()
        ans[case][i] = 1.0
    return ans


def judge_4_one_hot_multi_class_by_equal_2d(y_pred, y_target):
    """
    根据分类情况是否正确, 返回准确率, 适用于多分类问题.
    注意: 每行只有一个正确的值.
    """
    temp = y_target[y_pred == 1.0]
    return (temp.sum() / y_target.shape[0]).item()
