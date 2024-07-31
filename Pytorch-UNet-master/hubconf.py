import torch  # PyTorch 库，用于深度学习
from unet import UNet as _UNet  # 从 unet 模块导入 UNet 类，并重命名为 _UNet


# 定义函数创建并返回一个预训练的 UNet 模型
def unet_carvana(pretrained=False, scale=0.5):
    """
    UNet model trained on the Carvana dataset (https://www.kaggle.com/c/carvana-image-masking-challenge/data).
    Set the scale to 0.5 (50%) when predicting.
    """
    # 创建一个 UNet 模型实例，设置输入通道为 3，输出类别为 2，使用反卷积进行上采样
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)

    # 如果需要加载预训练权重
    if pretrained:
        # 根据缩放比例选择对应的预训练模型文件
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:
            # 如果缩放比例不是 0.5 或 1.0，则抛出异常
            raise RuntimeError('Only 0.5 and 1.0 scales are available')

        # 从指定 URL 加载预训练模型的权重
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)

        # 移除 state_dict 中的 'mask_values' 键（如果存在）
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')

        # 将加载的权重应用到模型
        net.load_state_dict(state_dict)

    # 返回创建的 UNet 模型
    return net
