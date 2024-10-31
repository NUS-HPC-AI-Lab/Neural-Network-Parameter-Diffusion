import os
import torch
import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 强制每次重新下载模型
model_name = "vit_tiny_patch16_224_dino"
model = timm.create_model(model_name, pretrained=True, num_classes=1000, checkpoint_path='')
model = model.to(device)
model.eval()

print(f"Loaded model: {model_name}")

# 获取模型的默认配置
config = model.default_cfg

# 创建验证集转换
val_transform = create_transform(
    input_size=config['input_size'],
    crop_pct=config['crop_pct'],
    interpolation=config['interpolation'],
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    is_training=False
)

print("Validation transform:", val_transform)

# 加载ImageNet验证集
imagenet_val_dir = "/home/wangkai/data/imagenet/val"  # 替换为你的ImageNet验证集路径
val_dataset = ImageFolder(root=imagenet_val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"Validation dataset size: {len(val_dataset)}")

# 评估函数
@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每100个批次显示一次当前准确率
        if (batch_idx + 1) % 100 == 0 or batch_idx == len(data_loader) - 1:
            current_accuracy = 100 * correct / total
            print(f"Batch {batch_idx + 1}/{len(data_loader)}, Current Accuracy: {current_accuracy:.2f}%")

    final_accuracy = 100 * correct / total
    return final_accuracy

# 在验证集上评估模型
final_accuracy = evaluate(model, val_loader)
print(f"Final Top-1 Accuracy on ImageNet validation set: {final_accuracy:.2f}%")