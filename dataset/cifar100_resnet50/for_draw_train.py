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
from torchvision.datasets import CIFAR100

try:  # relative import
    from model import create_model
except ImportError:
    from .model import create_model

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)

config = {
    "dataset_root": "from_additional_config",
    "batch_size": 128,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "seed": 42,
    "tag": os.path.basename(os.path.dirname(__file__)),
}
config.update(additional_config)


test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

test_dataset = CIFAR100(root=config["dataset_root"], train=False, download=True, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                         num_workers=config["num_workers"], pin_memory=True)

# Create model
model = create_model(num_classes=100)
model = model.to(device)

@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicts = []

    pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
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

    model.train()
    return loss, acc, all_targets, all_predicts
