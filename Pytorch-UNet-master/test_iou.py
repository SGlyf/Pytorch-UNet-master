import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

# 设置图像和掩码目录
dir_img = Path('./testdata_B/imgs/')
dir_mask = Path('./testdata_B/masks/')


# 定义 IoU 计算函数
def Iou(pred, true):
    intersection = pred * true
    temp = pred + true
    union = temp - intersection
    smooth = 1e-8
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score


# 保存预测图片
def save_predicted_images(images, masks_pred, batch_index, output_dir='output'):
    output_img_dir = os.path.join(output_dir, 'images')
    output_mask_dir = os.path.join(output_dir, 'masks')
    output_blend_dir = os.path.join(output_dir, 'blended')

    print("Saving images to:", output_img_dir)
    print("Saving masks to:", output_mask_dir)
    print("Saving blended images to:", output_blend_dir)

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        print("Created directory:", output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
        print("Created directory:", output_mask_dir)
    if not os.path.exists(output_blend_dir):
        os.makedirs(output_blend_dir)
        print("Created directory:", output_blend_dir)

    for i, (img, mask) in enumerate(zip(images, masks_pred)):
        print(f"Processing image {batch_index}_{i}")

        # 保存原图
        img_np = img.cpu().numpy().transpose(1, 2, 0) * 255
        img_pil = Image.fromarray(img_np.astype('uint8'))
        img_path = os.path.join(output_img_dir, f'image_{batch_index}_{i}.png')
        img_pil.save(img_path)
        print(f"Saved image: {img_path}")

        # 保存掩码
        mask_np = mask.cpu().numpy() * 255
        mask_pil = Image.fromarray(mask_np.astype('uint8'))
        mask_path = os.path.join(output_mask_dir, f'mask_{batch_index}_{i}.png')
        mask_pil.save(mask_path)
        print(f"Saved mask: {mask_path}")

        # 将掩码中的车道线变为红色
        mask_rgb = np.zeros_like(img_np)
        mask_rgb[:, :, 0] = mask_np  # Red channel
        mask_rgb = np.clip(mask_rgb, 0, 255)

        # 增强车道线颜色
        mask_rgb[:, :, 0] = np.where(mask_rgb[:, :, 0] > 0, 255, 0)

        # 将原图和掩码融合
        blended = cv2.addWeighted(img_np.astype('uint8'), 0.6, mask_rgb.astype('uint8'), 0.4, 0)
        blended_path = os.path.join(output_blend_dir, f'blended_{batch_index}_{i}.png')
        cv2.imwrite(blended_path, blended)
        print(f"Saved blended image: {blended_path}")


# 定义模型测试函数
def test_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        output_dir: str = 'output'  # 修改输出目录参数
):
    # 1. 创建数据集
    dataset = BasicDataset(Path(dir_img), Path(dir_mask), img_scale)

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)
    test_size = len(test_loader)

    # 初始化日志记录
    global_step = 0
    model.eval()
    iou_list = []
    with tqdm(total=test_size, unit='img') as pbar:
        for batch_index, batch in enumerate(test_loader):
            images, true_masks = batch['image'], batch['mask']

            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    sigmoid_pred = (F.sigmoid(masks_pred.squeeze(1)) > 0.5).float()
                    iou = Iou(sigmoid_pred, true_masks.float())
                    iou_list.append(torch.mean(iou, dim=0))
                    # 保存预测图片
                    save_predicted_images(images, sigmoid_pred, batch_index, output_dir)

            pbar.update(images.shape[0])
            global_step += 1
        print("mean iou: ", sum(iou_list) / len(iou_list))


# 定义命令行参数解析函数
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1.25e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default='D:\\desk\\Pytorch-UNet-master_new\\Pytorch-UNet-master\\sota_pth\\dataset_A_sota.pth',
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


# 主程序入口
if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 创建 UNet 模型
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    # 测试模型
    test_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        output_dir='output'  # 设置输出目录
    )
