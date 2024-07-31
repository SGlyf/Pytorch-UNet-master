import torch  # 导入 PyTorch 库
from torch import Tensor  # 从 PyTorch 导入 Tensor 类

# 计算单个或所有批次的 Dice 系数
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # 断言输入和目标张量的大小相同
    assert input.size() == target.size()
    # 断言输入张量的维度为3，或者 reduce_batch_first 为 False
    assert input.dim() == 3 or not reduce_batch_first

    # 根据输入张量的维度确定求和维度
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 计算交集部分，并乘以2
    inter = 2 * (input * target).sum(dim=sum_dim)
    # 计算输入和目标的和
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # 避免 sets_sum 为0的情况，确保分母不为0
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # 计算 Dice 系数
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()  # 返回平均 Dice 系数

# 计算多类别 Dice 系数
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # 对所有类别计算平均 Dice 系数
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

# 计算 Dice 损失
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice 损失在0到1之间，用于最小化目标
    fn = multiclass_dice_coeff if multiclass else dice_coeff  # 根据 multiclass 标志选择相应的 Dice 系数计算函数
    return 1 - fn(input, target, reduce_batch_first=True)  # 返回 Dice 损失
