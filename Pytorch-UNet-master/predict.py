import argparse  # 用于解析命令行参数
import logging  # 用于记录日志信息
import os  # 提供与操作系统交互的功能

import numpy as np  # 用于数组操作和数值计算
import torch  # PyTorch 库，用于深度学习
import torch.nn.functional as F  # 包含神经网络常用函数
from PIL import Image  # Python Imaging Library，用于图像处理
from torchvision import transforms  # 图像处理的转换操作

from utils.data_loading import BasicDataset  # 数据加载类
from unet import UNet  # UNet 模型类
from utils.utils import plot_img_and_mask  # 用于可视化图像和掩码的工具函数

# 预测图像的掩码
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()  # 设置模型为评估模式
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))  # 预处理图像
    img = img.unsqueeze(0)  # 增加 batch 维度
    img = img.to(device=device, dtype=torch.float32)  # 将图像加载到设备

    with torch.no_grad():  # 禁用梯度计算
        output = net(img).cpu()  # 前向传播并将输出移动到 CPU
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')  # 调整输出大小
        if net.n_classes > 1:
            mask = output.argmax(dim=1)  # 多类情况下，取最大概率的类
        else:
            mask = torch.sigmoid(output) > out_threshold  # 二分类情况下，应用阈值化

    return mask[0].long().squeeze().numpy()  # 返回处理后的掩码

# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')  # 模型文件
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)  # 输入图像文件
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')  # 输出掩码文件
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')  # 是否可视化处理图像
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')  # 是否不保存输出掩码
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')  # 掩码阈值
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')  # 图像缩放因子
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')  # 是否使用双线性上采样
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  # 类别数量

    return parser.parse_args()  # 解析并返回参数

# 生成输出文件名
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'  # 生成输出文件名

    return args.output or list(map(_generate_name, args.input))  # 如果未指定输出文件名，则自动生成

# 将掩码转换为图像
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

# 主程序入口
if __name__ == '__main__':
    args = get_args()  # 获取命令行参数
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  # 配置日志记录

    in_files = args.input  # 输入文件列表
    out_files = get_output_filenames(args)  # 输出文件列表

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)  # 创建 UNet 模型

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    logging.info(f'Loading model {args.model}')  # 记录加载模型信息
    logging.info(f'Using device {device}')  # 记录使用的设备

    net.to(device=device)  # 将模型移动到设备
    state_dict = torch.load(args.model, map_location=device)  # 加载模型参数
    mask_values = state_dict.pop('mask_values', [0, 1])  # 获取掩码值并从 state_dict 中删除
    net.load_state_dict(state_dict)  # 加载模型参数到模型

    logging.info('Model loaded!')  # 记录模型加载完成

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')  # 记录正在处理的图像
        img = Image.open(filename)  # 打开图像文件

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)  # 预测图像的掩码

        if not args.no_save:
            out_filename = out_files[i]  # 输出文件名
            result = mask_to_image(mask, mask_values)  # 将掩码转换为图像
            result.save(out_filename)  # 保存输出图像
            logging.info(f'Mask saved to {out_filename}')  # 记录保存信息

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')  # 记录正在可视化的图像
            plot_img_and_mask(img, mask)  # 可视化图像和掩码
