import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import random
import pdb
import time
import uuid
import sys
import timm
sys.path.append('../data_utils/')

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    description = "loss={:.4f} acc={:.2f}%"
    total = 0
    with tqdm(train_loader) as batchs:
        for idx, (data, target) in enumerate(batchs):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            loss = F.cross_entropy(output, target)

            total += data.shape[0]
            total_loss += loss.detach().item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            batchs.set_description(description.format(total_loss / total, 100*correct / total))

            loss.backward()
            optimizer.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
        .format(test_loss, acc))
    return acc, test_loss


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='resnet18', help='which model structure')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')

transform = transforms.Compose([transforms.ToTensor(),])
                            #    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

trainset = torchvision.datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

testset = torchvision.datasets.MNIST(root="./data/",
                        transform = transform,
                        train = False)

    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader = torch.utils.data.DataLoader(
trainset, batch_size=100, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(
testset, batch_size=100, shuffle=False, num_workers=0)

net = timm.create_model(args.model, pretrained=True)

num_params = sum([np.prod(p.size()) for p in (net.parameters())])
print(f'parameters number is {num_params}')
noisy_accs = {}
for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0 ]:
        noisy_accs[alpha] = 0

clean_accs = []
net = timm.create_model('resnet18', pretrained=False)



optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4) 

num_channels = 1  # 单通道输入
# net.reset_classifier(num_classes=10, global_pool='',pool_type='conv')  # 重置分类器
net.conv1 = nn.Conv2d(num_channels, net.conv1.out_channels, kernel_size=net.conv1.kernel_size, stride=net.conv1.stride, padding=net.conv1.padding)
net=net.cuda()
for e in range(20):
    train(net, trainloader,optimizer)
acc, loss = test(net.cuda(), testloader)
print(f'=====Normal accuracy is {acc}=====')

clean_accs.append(acc)
i = 0
for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0 ]:
## add noise
    print(f'<<<<<<<<<<<<<<<<<<<<<<<The alpha is {alpha}>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    noise_std = 1.0
    # net = timm.create_model('resnet50', pretrained=True)
    # net.conv1 = nn.Conv2d(num_channels, net.conv1.out_channels, kernel_size=net.conv1.kernel_size, stride=net.conv1.stride, padding=net.conv1.padding)

    # net=net.cuda()

    for param in net.parameters():
        if param.requires_grad:
            # print(param.shape)
            noise = torch.randn(param.size(), device = device) * noise_std
            param.data =  (1-alpha) * param.data + alpha * noise
    noisy_acc, loss = test(net.cuda(), testloader)
    noisy_accs[alpha] += noisy_acc


print(f'parameters number is {num_params}')
print('Final noisy accs is', noisy_accs)
print('Natural accs is', np.mean(np.array(clean_accs)))

            