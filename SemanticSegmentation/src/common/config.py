"""
在这个类中完成对一些基础设置的配置.
"""
import os
import torch


class Config(object):
    """在这个类中, 完成对基础属性的配置."""
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    src_dir = os.path.join(project_dir, 'src')
    data_dir = os.path.join(project_dir, 'data')
    out_dir = os.path.join(project_dir, 'out')

    @classmethod
    def join_path(cls, dirname: str, basename):
        target_dir = getattr(cls, '{}_dir'.format(dirname))
        return os.path.join(target_dir, basename.replace('/', '\\'))


class MNISTConfig(Config):
    """数字识别的数据集的具体配置."""
    batch_size = 256
    lr = 0.005
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
