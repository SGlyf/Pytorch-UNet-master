import torch  # PyTorch 库，用于深度学习
import torch.nn.functional as F  # 包含神经网络常用函数
from tqdm import tqdm  # 进度条显示

from utils.dice_score import multiclass_dice_coeff, dice_coeff  # 导入 Dice 系数计算函数

# 定义 IoU 计算函数
def Iou(pred, true):
    intersection = pred * true  # 计算交集 pred ∩ true
    temp = pred + true  # 计算 pred + true
    union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8  # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)  # 计算 IoU
    return iou_score  # 返回 IoU 分数

# 使用 @torch.inference_mode() 装饰器，禁用梯度计算以提高推理速度和节省内存
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()  # 将模型设置为评估模式
    num_val_batches = len(dataloader)  # 验证集的批次数量
    dice_score = 0  # 初始化 Dice 分数
    iou_score = 0  # 初始化 IoU 分数

    # 使用自动混合精度计算
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, desc='Validation round'):  # 迭代验证集，并显示进度条
            image, mask_true = batch['image'], batch['mask']  # 获取图像和真实掩码
            print("start eval")  # 打印开始评估信息

            # 将图像和掩码移动到正确的设备和数据类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 预测掩码
            mask_pred = net(image)

            if net.n_classes == 1:  # 如果是二分类问题
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'  # 验证掩码值
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()  # 应用 Sigmoid 函数并进行阈值化
                # 计算 Dice 分数
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_score += Iou(mask_pred, mask_true)  # 计算 IoU 分数

            else:  # 如果是多分类问题
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['  # 验证掩码值
                # 转换为 one-hot 格式
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # 计算 Dice 分数，忽略背景
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()  # 将模型设置回训练模式
    return dice_score / max(num_val_batches, 1), iou_score / max(num_val_batches, 1)  # 返回平均 Dice 分数和 IoU 分数
