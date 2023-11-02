'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import json

from models import *
from utils import progress_bar,get_testdata,get_traindata


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--save_layer', default='bn2', type=str, help='save_layer bn')
parser.add_argument('--num_model', default=500, help='model numbers')
parser.add_argument('--gpu', default=0, help='model numbers')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_acc = 0
train_epoch = 1
all_epoch = train_epoch+int(args.num_model)

# Data
print('==> Preparing data..')

trainloader = get_traindata('cifar100')
testloader = get_testdata('cifar100')

net = ResNet18(100)

for name,weights in net.named_parameters():
    print(name,weights.size())
print('save_layers',args.save_layer)
if args.save_layer=='fc':
    train_layers = ['module.linear.weight','module.linear.bias']
elif args.save_layer=='conv4_1_2':
    train_layers = ['layer4.1.conv2.weight']
elif args.save_layer=='bn2':
    train_layers = ['layer4.1.bn1.weight','layer4.1.bn1.bias','layer4.1.bn2.weight','layer4.1.bn2.bias']
elif args.save_layer=='bn':
    train_layers = []
    for name,module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            
            for name_sub,weights in module.named_parameters():
                train_layers.append(name+'.'+name_sub)
                print(name+'.'+name_sub,weights.shape)
print("train_layers",train_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def train_eval(epoch):
    print('\nEpoch: %d' % epoch)
    net.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return 100.*correct/total
def fix_partial_model(train_list,net):
    print(train_list)
    for name,weights in net.named_parameters():
        if name not in train_list:
            weights.requires_grad = False
def save_part(train_list,net):
    part_param = []
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param.append(weights.detach().cpu())
    return part_param

net = net.to(device)
parameters = []
save_dir = os.path.join('parameters/', "ResNet18_cifar100")
os.makedirs(save_dir, exist_ok=True)
full_model_path = os.path.join(save_dir, args.save_layer+"full_model.pkl")
part_model_folder = os.path.join(save_dir, args.save_layer)
os.makedirs(part_model_folder, exist_ok=True)


#=======================save train traget=====================
for epoch in range(0, all_epoch):
    train(epoch)
    test()
    if epoch == (train_epoch-1):
        torch.save(net.state_dict(), full_model_path)
        fix_partial_model(train_layers,net)
        parameters = []
    if epoch >= train_epoch:
        parameters.append(save_part(train_layers,net))
        if len(parameters) == 10:
            torch.save(parameters, os.path.join(part_model_folder, "parameters_{}.pt".format(epoch)))
            parameters = []
    scheduler.step()
