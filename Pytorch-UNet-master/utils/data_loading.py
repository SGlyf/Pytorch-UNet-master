import logging  # 用于记录日志信息
import numpy as np  # 用于数组操作和数值计算
import torch  # PyTorch 库，用于深度学习
from PIL import Image  # Python Imaging Library，用于图像处理
from functools import lru_cache, partial  # 缓存和部分应用函数
from itertools import repeat  # 用于重复迭代
from multiprocessing import Pool  # 多进程处理
from os import listdir  # 用于列出目录内容
from os.path import splitext, isfile, join  # 用于处理文件路径
from pathlib import Path  # 用于处理路径对象
from torch.utils.data import Dataset  # 数据集类
from tqdm import tqdm  # 进度条显示

# 加载图像，根据文件扩展名选择适当的加载方式
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))  # 从 .npy 文件加载
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())  # 从 .pt 或 .pth 文件加载
    else:
        return Image.open(filename)  # 从其他图像文件加载

# 获取唯一的掩码值，用于确定掩码的不同类别
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]  # 找到掩码文件
    mask = np.asarray(load_image(mask_file))  # 加载掩码
    if mask.ndim == 2:
        return np.unique(mask)  # 如果掩码是二维的，返回唯一值
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)  # 如果掩码是三维的，返回唯一值
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')  # 如果维度不正确，抛出异常

# 定义基本数据集类
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_bin'):
        self.images_dir = Path(images_dir)  # 图像目录
        self.mask_dir = Path(mask_dir)  # 掩码目录
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'  # 确保缩放比例在0到1之间
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')  # 如果没有找到输入文件，抛出异常

        logging.info(f'Creating dataset with {len(self.ids)} examples')  # 记录数据集创建信息
        logging.info('Scanning mask files to determine unique values')  # 记录掩码扫描信息
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))  # 获取唯一掩码值
        logging.info(f'Unique mask values: {self.mask_values}')  # 记录唯一掩码值

    def __len__(self):
        return len(self.ids)  # 返回数据集大小

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)  # 计算新的宽度和高度
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'  # 确保新尺寸有效
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)  # 调整图像大小
        img = np.asarray(pil_img)  # 转换为数组

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask  # 返回处理后的掩码
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0  # 归一化图像
            return img  # 返回处理后的图像
    # 图像做归一化，maks做二值化
    def __getitem__(self, idx):
        name = self.ids[idx]  # 获取样本 ID
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))  # 获取掩码文件
        img_file = list(self.images_dir.glob(name + '.*'))  # 获取图像文件

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'  # 确保有且仅有一个图像文件
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'  # 确保有且仅有一个掩码文件
        mask = load_image(mask_file[0])  # 加载掩码
        img = load_image(img_file[0])  # 加载图像

        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'  # 确保图像和掩码尺寸相同

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)  # 预处理图像
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)  # 预处理掩码

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),  # 返回图像
            'mask': torch.as_tensor(mask.copy()).long().contiguous()  # 返回掩码
        }

# 定义 Carvana 数据集类，继承自 BasicDataset
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')  # 调用父类构造函数
