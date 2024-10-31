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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
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
        "model_name": "vit_base_patch16_224"
    }
    config.update(additional_config)
    print("Final config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    return config


def get_data_loaders(config):
    print(f"Initializing data loaders with dataset_root: {config['dataset_root']}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_path = os.path.join(config['dataset_root'], 'train')
    val_path = os.path.join(config['dataset_root'], 'val')

    print(f"Checking if train path exists: {os.path.exists(train_path)}")
    print(f"Checking if val path exists: {os.path.exists(val_path)}")

    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    test_dataset = ImageFolder(root=val_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_optimizer_and_scheduler(model, config, freeze_all_except_last_two_norms=True):
    if freeze_all_except_last_two_norms:
        trainable_params = []
        norm_layers = [name for name, _ in model.named_modules()]
        # import pdb; pdb.set_trace()
        last_two_norms = ['blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'norm.weight', 'norm.bias']
        for name, param in model.named_parameters():
            # if any(norm in name for norm in last_two_norms):
            # print(name)
            # print(any(norm in name for norm in last_two_norms))
            if name in last_two_norms:
                param.requires_grad = True
                trainable_params.append(param)
                import pdb; pdb.set_trace()
            else:
                param.requires_grad = False
        import pdb;
        pdb.set_trace()
        optimizer = optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(get_data_loaders(config)[0]),
                                               eta_min=config["save_learning_rate"])
    return optimizer, scheduler


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


def save_checkpoint(model, batch_idx, acc, config):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    last_two_norms = ['blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'norm.weight', 'norm.bias']
    save_state = {key: value.cpu().to(torch.float32) for key, value in model.state_dict().items()
                  if any(norm in key for norm in last_two_norms)}
    save_path = f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{config['seed']:04d}_{config['tag']}.pth"
    torch.save(save_state, save_path)
    print(f"Saved: {save_path}")


def finetune(config):
    print("Starting finetune process")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(config['seed'])
    print(f"Seed set to {config['seed']}")

    train_loader, test_loader = get_data_loaders(config)

    print(f"Loading model: {config['model_name']}")
    # import pdb; pdb.set_trace()
    model = timm.create_model(config['model_name'], pretrained=True, num_classes=1000)
    model = model.to(device)

    optimizer, scheduler = get_optimizer_and_scheduler(model, config, freeze_all_except_last_two_norms=True)

    print("Initial test:")
    model.eval()
    test(model, test_loader, device)

    total_batches = len(train_loader)
    save_interval = max(1, total_batches // config["total_save_number"])
    print(f"Total batches: {total_batches}, Save interval: {save_interval}")

    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm(train_loader, desc='Training', ncols=100)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if (batch_idx + 1) % save_interval == 0 or batch_idx == total_batches - 1:
            loss, acc, _, _ = test(model, test_loader, device)
            save_checkpoint(model, batch_idx, acc, config)

        pbar.set_postfix({'Loss': f'{loss:.3f}'})

    print("Fine-tuning completed.")


if __name__ == '__main__':
    config = get_config()
    config['dataset_root'] = "/home/wangkai/data/imagenet"
    # import pdb; pdb.set_trace()
    finetune(config)
