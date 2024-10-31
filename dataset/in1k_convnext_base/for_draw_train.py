import os
import random
import numpy as np
import json
import warnings
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm

# try:  # relative import
#     from model import create_model
# except ImportError:
#     from .model import create_model

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
print(f"Loading config from: {config_file}")
with open(config_file, "r") as f:
    additional_config = json.load(f)

config = {
    "dataset_root": os.path.expanduser("/home/wangkai/data/imagenet"),
    "batch_size": 128,
    "num_workers": 4,
    "learning_rate": 0.05,
    "weight_decay": 5e-4,
    "epochs": 1,
    "save_learning_rate": 0.05,
    "total_save_number": 200,
    "tag": os.path.basename(os.path.dirname(__file__)),
    "freeze_epochs": 0,
    "seed": 40,
    "model_name": "convnext_base"
}
config.update(additional_config)
print("Final config:")
for key, value in config.items():
    print(f"{key}: {value}")

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

config['dataset_root'] = "/home/wangkai/data/imagenet"
val_path = os.path.join(config['dataset_root'], 'val')

print(f"Checking if val path exists: {os.path.exists(val_path)}")

test_dataset = ImageFolder(root=val_path, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                         num_workers=config["num_workers"], pin_memory=True)

print(f"Test dataset size: {len(test_dataset)}")

model = timm.create_model(config['model_name'], pretrained=True, num_classes=1000)
model = model.to(device)



@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicts = []

    pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        all_targets.extend(targets.cpu().tolist())
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.cpu().tolist())
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()

        pbar.set_postfix({'Loss': f'{test_loss / (pbar.n + 1):.3f}', 'Acc': f'{100. * correct / total:.2f}%'})

    loss = test_loss / len(test_loader)
    acc = correct / total
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    return loss, acc, all_targets, all_predicts

