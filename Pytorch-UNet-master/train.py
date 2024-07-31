import argparse
import logging
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from bisect import bisect_left
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
# argparse：解析命令行参数。
# logging：记录日志。
# os：操作系统相关功能。
# tensorboardX：用于TensorBoard可视化。
# torch 和 torch.nn：用于定义和训练神经网络。
# torchvision.transforms：用于图像数据的预处理。
# pathlib.Path：操作路径。
# tqdm：显示训练进度条。
# bisect_left：二分查找，用于学习率调度。
# unet、BasicDataset、CarvanaDataset、dice_loss：自定义模块，用于模型、数据加载和损失计算


dir_img = Path('./dataset_B/imgs/')
dir_mask = Path('./dataset_B/masks/')
dir_checkpoint = Path('./checkpoints/')
# 这三行代码定义了图像和掩码数据的路径以及检查点的保存路径。

class DetailNet_LRScheduler(object):
    # 学习率调度器用于在训练过程中根据预定义的学习率步骤
    #  self.optimizer, [20000, 40000, 60000], [lr, lr/3, lr/9, lr/27]
    def __init__(self, optimizer, lr_steps, lrs, last_iter=0):
        # optimizer：优化器对象，用于更新模型的参数
        # lr_steps：一个整数列表，表示在训练过程中学习率变化的步骤
        # lrs：一个浮点数列表，表示在对应步骤中使用的学习率
        # last_iter：上次迭代的步数，默认为0
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        # 验证optimizer参数是否为 torch.optim.Optimizer 的实例。如果不是，则抛出一个 TypeError 异常。
        self.optimizer = optimizer
        if last_iter == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        #         初始化 optimizer，并处理优化器的初始学习率\如果 last_iter 为0（表示从头开始训练），则为每个参数组设置初始学习率。
        # 如果 last_iter 不为0（表示从中断处恢复训练），则检查每个参数组是否包含初始学习率
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter
        # 这段代码保存优化器中每个参数组的初始学习率，并记录上次迭代的步数。

        assert len(lr_steps) + 1 == len(lrs), "{} vs {}".format(lr_steps, lrs)
        for x in lr_steps:
            assert isinstance(x, int)
        if not list(lr_steps) == sorted(lr_steps):
            raise ValueError('lr_steps should be a list of'
                             ' increasing integers. Got {}', lr_steps)
        self.lr_steps = lr_steps
        self.lrs = lrs
    # 确保 lr_steps 的长度比 lrs 的长度少1。
    # 确保 lr_steps 中的每个元素都是整数。
    # 确保 lr_steps 是按升序排列的列表。

    def _get_new_lr(self):
        # 在lr_steps中寻找last_iter合适的插入位置，左侧插入
        pos = bisect_left(self.lr_steps, self.last_iter)
        # 返回对应位置的学习率
        return [self.lrs[pos]]

    def get_lr(self):
        # 获取当前优化器中每个参数组的学习率
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        # 如果未提供新的迭代步数，则将其设为self.last_iter + 1
        if this_iter is None:
            this_iter = self.last_iter + 1
        # 更新当前的迭代步数
        self.last_iter = this_iter
        # 遍历优化器的参数组和对应的新学习率
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            # 将新学习率赋值给参数组
            param_group['lr'] = lr


def Iou(pred,true):
    intersection = pred * true          # 计算交集  pred ∩ true
    temp = pred + true                  # pred + true
    union = temp - intersection         # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8                       # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score
# 这个函数计算交并比（IoU），用于评估模型预测与真实标签的重叠情况。

def train_model(
        model,
        device,  # 训练时使用的设备（例如，torch.device('cuda') 或 torch.device('cpu')）
        epochs: int = 5,  # 训练的轮次数量，默认值为5
        batch_size: int = 1,  # 每次训练的批次大小，默认值为1
        learning_rate: float = 1e-5,  # 学习率，控制模型权重更新的步长，默认值为1e-5
        val_percent: float = 0.1,  # 用于验证的数据集比例，默认值为0.1（即10%的数据用于验证）
        save_checkpoint: bool = True,  # 是否在每个epoch结束时保存模型检查点，默认值为True
        img_scale: float = 0.5,  # 图像缩放比例，控制输入图像的尺寸，默认值为0.5
        amp: bool = False,  # 是否使用自动混合精度（Automatic Mixed Precision），可以减少显存占用并加速训练，默认值为False
        weight_decay: float = 1e-8,  # 权重衰减系数，用于L2正则化，防止过拟合，默认值为1e-8
        momentum: float = 0.999,  # 用于加速梯度下降法收敛，默认值为0.999
        gradient_clipping: float = 1.0  # 梯度裁剪的阈值，用于防止梯度爆炸，默认值为1.0
):
    # 1. 创建数据集
    dataset = BasicDataset(Path(dir_img), Path(dir_mask), img_scale)

    # 2. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    train_size = len(train_loader)

    # 3. 初始化日志记录
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    # 在训练开始时记录一些基本信息，包括训练轮次、批次大小、学习率、是否保存检查点、使用的设备、图像缩放比例和是否使用混合精度

    # 4. 设置优化器、损失函数、学习率调度器和自动混合精度缩放
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = DetailNet_LRScheduler(optimizer, [7500, 15000, 18700],
                                      [learning_rate, learning_rate / 2, learning_rate / 4,
                                       learning_rate / 8])  # 目标是最大化Dice得分
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    writer = SummaryWriter('logs')

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        step = 0
        with tqdm(desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 将图像和掩码数据转移到指定设备
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 使用自动混合精度训练
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # 计算损失并更新模型参数
                        loss = 8.0 * criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        sigmoid_pred = (F.sigmoid(masks_pred.squeeze(1)) > 0.5).float()
                        iou = Iou(sigmoid_pred, true_masks.float())
                        logging.info('iou_score: {}'.format(iou))
                    else:
                        # 计算损失并更新模型参数
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        iou = Iou(F.softmax(masks_pred, dim=1).float(),
                                  F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float())
                        logging.info('iou_score: {}'.format(iou))

                # 记录损失、IoU和学习率
                writer.add_scalar('total_loss', loss, global_step=global_step)
                writer.add_scalar('iou_score', iou, global_step=global_step)
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=global_step)

                # 优化器梯度归零
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                scheduler.step()
                grad_scaler.update()

                # 更新进度条和全局步骤
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # 保存检查点
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')  #训练的轮次数
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size') #每个训练批次的样本数量
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,  #学习率
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=r'D:\desk\Pytorch-UNet-master_new\Pytorch-UNet-master\sota_pth\dataset_B_sota.pth', help='Load model from a .pth file')   #从指定的 .pth 文件加载模型
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images')  #图像的缩放为原来的25%
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')  #用于验证的数据比例（0-100）
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')  #是否使用混合精度训练
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')  #是否使用双线性上采样。否则使用反卷积（转置卷积）上采样。
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')  #输出的类别数量（即分割的类别数）

    return parser.parse_args()


if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()

    # 设置日志记录的基本配置
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # 确定使用的设备（如果有GPU则使用GPU，否则使用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 根据您的数据修改此处
    # n_channels=3表示RGB图像
    # n_classes表示每个像素的概率数量
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # 将模型转换为通道优先的内存格式
    model = model.to(memory_format=torch.channels_last)
    # 计算模型参数的总数量
    parameters_num = sum(x.numel() for x in model.parameters())
    print("U-net have {}M parameters in total".format(parameters_num / 1e6))
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 如果提供了模型权重文件，则加载模型权重
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']  # 删除不需要的键
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    # 将模型移动到指定设备
    model.to(device=device)
    try:
        # 开始训练模型
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        # 处理显存不足的错误
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()  # 清空缓存
        model.use_checkpointing()  # 启用检查点保存以减少内存使用
        # 再次尝试训练模型
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
