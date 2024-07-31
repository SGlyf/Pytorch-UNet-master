""" Parts of the U-Net model """

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的函数式模块


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    # 定义一个双重卷积层，即两次 (卷积 => 批量归一化 => ReLU)

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()  # 调用父类的构造函数
        if not mid_channels:
            mid_channels = out_channels  # 如果未提供中间通道数，则设置为输出通道数
        # 定义一个包含两次卷积、批量归一化和 ReLU 激活函数的顺序层
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)  # 前向传播时，输入通过双重卷积层


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # 定义一个下采样层，包括最大池化层和双重卷积层

    def __init__(self, in_channels, out_channels):
        super().__init__()  # 调用父类的构造函数
        # 定义一个包含最大池化层和双重卷积层的顺序层
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 的最大池化层
            DoubleConv(in_channels, out_channels)  # 双重卷积层
        )

    def forward(self, x):
        return self.maxpool_conv(x)  # 前向传播时，输入通过最大池化层和双重卷积层


class Up(nn.Module):
    """Upscaling then double conv"""
    # 定义一个上采样层，包括上采样操作和双重卷积层

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()  # 调用父类的构造函数

        # 如果使用双线性插值，则用正常卷积来减少通道数
        if (bilinear):
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样输入
        # 输入的尺寸是 (通道, 高, 宽)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 填充上采样后的张量，使其尺寸与跳跃连接的张量一致
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 如果有填充问题，请参阅以下链接
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)  # 在通道维度上连接跳跃连接的张量和上采样后的张量
        return self.conv(x)  # 通过双重卷积层处理连接后的张量


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()  # 调用父类的构造函数
        # 定义一个 1x1 的卷积层，用于生成最终的输出
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  # 前向传播时，输入通过 1x1 卷积层
