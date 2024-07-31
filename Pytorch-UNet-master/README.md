# 安装环境
pip install -r requirments.txt
pip install torch>=1.8.1

# 使用的网络结构为 U-net，https://blog.csdn.net/weixin_55073640/article/details/123060574

# 使用的损失函数为DICE loss， https://blog.csdn.net/BluErroR/article/details/134541671


# 如何运行
# 首先将训练集中的图片 复制到data/imgs下，再将训练集中label图片 复制到data/masks下
# python train.py 即可开始训练
# python test_iou.py 测试验证集中的iou指标


# /predict.py 用于把预测mask保存下来，使用方法如下
 python predict.py -m /root/autodl-tmp/Pytorch-UNet-master/checkpoints/checkpoint_epoch300.pth -i /root/autodl-tmp/Pytorch-UNet-master/testdata_A/imgs/170927_063812587_Camera_6.jpg /root/autodl-tmp/Pytorch-UNet-master/testdata_A/imgs/170927_063823355_Camera_5.jpg -c 1

 # evaluate.py 中存放的是评估模型的指标，有iou和dice score，可以不用这个py文件，评估iou的指标已经放在test_iou.py文件夹中

 # utils/data_loading.py 中是 数据集加载的方法
 # /unet/文件夹下放的是，模型构建的代码

 # /logs 文件下记录了模型的训练过程中loss和iou score的变化，需要安装tensorboardX（pip install tensorboardX ）,在训练代码中已经写好了这块
 # 训练完成后，在命令行输入  tensorboard --log_dir logs/ ，即可查看loss和iou score变化的曲线图


 # others：如何切换不同的任务的训练集和测试集
 # 在train.py和test_iou.py中，修改下列文件夹路径（文件夹路径对应不同任务的数据集即可）
dir_img = Path('./dataset_A/imgs/')
dir_mask = Path('./dataset_A/masks/')
dir_checkpoint = Path('./checkpoints/')

# /sota_pth 下存放的是datasetA和datasetB两个训练集的最佳权重，到时候读取权重从这边文件读取就行