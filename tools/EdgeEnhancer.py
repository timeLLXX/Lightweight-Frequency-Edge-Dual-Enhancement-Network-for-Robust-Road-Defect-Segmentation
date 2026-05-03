import torch.nn as nn

class EdgeEnhancer(nn.Module):  # 边缘增强模块
    def __init__(self, in_dim, norm, act):  # 初始化函数，接收输入维度、归一化层和激活函数
        super().__init__()  # 调用父类构造函数
        self.out_conv = nn.Sequential(  # 定义输出卷积层
            nn.Conv2d(in_dim, in_dim, 1, bias=False),  # 1x1卷积，不使用偏置

            norm(in_dim),  # 归一化层
            # nn.Sigmoid()  # Sigmoid激活函数  : 将输出限制在0 到1 的范围内，有助于将增强的边缘特征与原始图像合并时保持一定的平衡，防止过强的增强导致信息丢失。
        )
        # 定义多脉冲激活函数
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # 定义平均池化层

    def forward(self, x):  # 前向传播函数
        """
            首先经过平均池化操作，这会平滑图像并降低细节。
            然后，通过计算输入图像与池化结果之间的差异（edge = x - edge），可以提取出图像的边缘信息。
            边缘通常是图像中像素值变化较大的地方，因此这种差异计算有助于强调边缘特征。
        """
        edge = self.pool(x)  # 对输入进行池化操作
        edge = x - edge  # 计算边缘信息，提取出图像的边缘信息
        edge = self.out_conv(edge)  # 通过输出卷积层处理边缘信息
        # 【通过残差 强化细节】
        return x + edge