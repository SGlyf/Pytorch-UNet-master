import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图


# 定义函数 plot_img_and_mask，用于绘制图像和对应的掩码
def plot_img_and_mask(img, mask):
    classes = mask.max() + 1  # 计算掩码中类别的数量（假设类别标签是连续的从0开始的整数）

    fig, ax = plt.subplots(1, classes + 1)  # 创建一个包含多个子图的图像窗口，子图数量为 classes + 1

    ax[0].set_title('Input image')  # 设置第一个子图的标题为 'Input image'
    ax[0].imshow(img)  # 在第一个子图中显示输入图像

    # 遍历所有类别，为每个类别绘制一个掩码子图
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')  # 设置每个掩码子图的标题
        ax[i + 1].imshow(mask == i)  # 在对应的子图中显示掩码（显示该类别的掩码部分）

    plt.xticks([]), plt.yticks([])  # 隐藏所有子图的刻度
    plt.show()  # 显示图像窗口
