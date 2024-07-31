""" Full assembly of the parts to form the complete network """

from .unet_parts import *  # 从当前模块的 unet_parts 文件中导入所有内容


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出通道数（类别数）
        self.bilinear = bilinear  # 是否使用双线性插值

        # 定义 U-Net 各层，使用从 unet_parts 中导入的模块
        self.inc = DoubleConv(n_channels, 64)  # 初始卷积层
        self.down1 = Down(64, 128)  # 下采样层 1
        self.down2 = Down(128, 256)  # 下采样层 2
        self.down3 = Down(256, 512)  # 下采样层 3
        factor = 2 if bilinear else 1  # 如果使用双线性插值，则缩小因子为2，否则为1
        self.down4 = Down(512, 1024 // factor)  # 下采样层 4
        self.up1 = Up(1024, 512 // factor, bilinear)  # 上采样层 1
        self.up2 = Up(512, 256 // factor, bilinear)  # 上采样层 2
        self.up3 = Up(256, 128 // factor, bilinear)  # 上采样层 3
        self.up4 = Up(128, 64, bilinear)  # 上采样层 4
        self.outc = OutConv(64, n_classes)  # 输出卷积层

    def forward(self, x):
        x1 = self.inc(x)  # 初始卷积层
        x2 = self.down1(x1)  # 下采样层 1
        x3 = self.down2(x2)  # 下采样层 2
        x4 = self.down3(x3)  # 下采样层 3
        x5 = self.down4(x4)  # 下采样层 4
        x = self.up1(x5, x4)  # 上采样层 1，输入为 x5 和 x4 的跳跃连接
        x = self.up2(x, x3)  # 上采样层 2，输入为上一步的输出和 x3 的跳跃连接
        x = self.up3(x, x2)  # 上采样层 3，输入为上一步的输出和 x2 的跳跃连接
        x = self.up4(x, x1)  # 上采样层 4，输入为上一步的输出和 x1 的跳跃连接
        logits = self.outc(x)  # 输出卷积层
        return logits  # 返回最终的预测结果

    def use_checkpointing(self):
        # 使用 PyTorch 的检查点功能，减少内存使用
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
