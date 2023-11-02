"""
本文件进行bn层敏感度测试
包括：噪声扰动敏感度 和 交叉混合敏感度测试
"""
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
sys.path.append('../../lib/')
from models_cifar import *

def load_params(params, model):
    ## params是一个列表形式存储的model.parameters()
    layer_idx = 0
    for p in model.parameters(): 
        p.data = params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

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

def get_dataset():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../../../datasets', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
        batch_size=256, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../../../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader

def load_param(param_list, bn_stats, model):

    layer_idx = 0
    for p in model.parameters(): 
        p.data = param_list[layer_idx]
        p.data.to(torch.device('cuda'))
        # p.data.requires_grad_(True)
        layer_idx += 1
    bn_stats_idx = 0
    # pdb.set_trace()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean = bn_stats[bn_stats_idx][0].cuda()
            module.running_var = bn_stats[bn_stats_idx][1].cuda()
            bn_stats_idx += 1
    # pdb.set_trace()
    return model

if __name__ == '__main__':



    best_acc = 0  # best test accuracy
    # trainloader, testloader = get_dataset()
    import os;
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    parameters = []
    bn_running_stats = []


    net = ResNet18_cifar10().cuda()
    count = 0
    count_bn = 0

    #下面这种方法会漏掉 shortcut
    # for name, param in net.named_parameters():
    #     print(name, param.requires_grad) #是有bn的其实
    #     print(param.numel())

    #     count += 1
    #     if 'bn' in name:
    #         count_bn += 1
    #     if 'shortcut' in name:
    #         print('check==========')
    #         print(count)
    # print(count)     #62
    # print(count_bn)  #34
    def find_bn_layers(model):
        bn_layers = []
        bn2d = []
        bn1d = []
        param_num = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layers.append((name, module))
      
                if hasattr(module, 'weight'):
                    param_num += torch.numel(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    param_num += torch.numel(module.bias)
        print(f'bn stats total number is {param_num}')
        return bn_layers  
    

    bns= find_bn_layers(net)
    print(len(bns))#20
    print('====all bn :===============')
    for bn in bns:
        print(bn[0])


def load_parameters(net, param):
    layer_idx = 0
    # print(len(param))
    for p in net.parameters(): 
        p.data = param[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return net

def save_bn(model):
    bn_param = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
                bn_param.append(module.weight.data.clone())
                bn_param.append(module.bias.data.clone())

    return bn_param

def hybrid(model1, model2, testloader, layer):
    acc, loss = test(model1.cuda(), testloader)
    acc, loss = test(model2.cuda(), testloader)

    if layer == 'bn':
        bn_param = []
        for name, module in model1.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                    bn_param.append(module.weight.data.clone())
                    bn_param.append(module.bias.data.clone())
        
        id=0
        for name, module in model2.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                    module.weight.data = bn_param[id]
                    module.bias.data = bn_param[id]
                    id += 2
        acc, loss = test(model2.cuda(), testloader)
    elif layer == 'fc':
        fc_param = []
        for name, param in model1.named_parameters():
            if 'linear' in name:
                    fc_param.append(param)
        print(len(fc_param))
        id = 0
        for name, param2 in model2.named_parameters():
            if 'linear' in name:
                    param2.data = fc_param[id]
                    id += 1
        acc, loss = test(model2.cuda(), testloader)
         


def pertub(test_number, param_ori, testloader, noisyrange, layer):
    clean_accs = []
    noisy_accs = {}
    improve = {}
    best_noisy_accs = {}

    for alpha in noisyrange:
        noisy_accs[alpha] = 0
        improve[alpha] = 0
        best_noisy_accs[alpha] = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18_cifar10().cuda()

    for test_id in range(test_number):
                net = load_parameters(net, param_ori)
                acc, loss = test(net.cuda(), testloader)
                # print(f'=====Normal accuracy is {acc}=====')

                clean_accs.append(acc)
                tmp = param_ori
                i = 0
                # total += 1
                for alpha in noisyrange:
                    ## add noise
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<The alpha is {alpha}>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    noise_std = 1.0
                    net = ResNet18_cifar10().cuda()

                    net = load_parameters(net, tmp).cuda()
                    
                    if layer == 'bn':
                        for name, module in net.named_modules():
                            if isinstance(module, nn.BatchNorm2d):
                                    noise = torch.randn(module.weight.data.size(), device = device) * noise_std
                                    module.weight.data  =  (1-alpha) * module.weight.data + alpha * noise
                                    noise = torch.randn(module.bias.data.size(), device = device) * noise_std
                                    module.bias.data =  (1-alpha) * module.bias.data + alpha * noise
                    elif layer == 'fc':
                        for name, param in net.named_parameters():
                            if 'linear' in name:
                                    noise = torch.randn(param.data.size(), device = device) * noise_std
                                    param.data  =  (1-alpha) * param.data + alpha * noise
                  
                    noisy_acc, loss = test(net.cuda(), testloader)
                    if noisy_acc > best_noisy_accs[alpha]:
                         best_noisy_accs[alpha] = noisy_acc
                    if noisy_acc > acc:
                        improve[alpha] += 1
                    noisy_accs[alpha] += noisy_acc
                    i += 1
    # print(len(noisy_accs))
    # print(len(noisy_accs[0]))
    for alpha in noisyrange:
        noisy_accs[alpha] /= test_number
        improve[alpha] /= test_number


    print('Final noisy accs is', noisy_accs)
    print('improve accs ratio is', improve)
    print('best noisy accs is', best_noisy_accs)
    print('Natural accs is', np.mean(np.array(clean_accs)))


if __name__ == '__main__':
    _, testloader = get_dataset()
    path = '../../parameters/cifar_res18'
    mode = 'per'
    layer = 'bn'
    if mode == 'per':
        files = os.listdir(path)[0]
        good = torch.load(path + '/' + files)[0]
        pertub(50,good,testloader,noisyrange=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ],layer=layer)

    elif mode == 'hy':
        files = os.listdir(path)
        good1 = torch.load(path + '/' + files[0])[0]
        good2 = torch.load(path + '/' + files[8])[0]
        model1, model2 = ResNet18_cifar10().cuda(), ResNet18_cifar10().cuda()
        model1 = load_parameters(model1, good1)
        model2 = load_parameters(model2, good2)
        hybrid(model1, model2, testloader,'fc')


"""
param1 = torch.load('parameters_2e2de224-2.pt')
param2 = torch.load('parameters_7fb376ee-2.pt')
parameters1=param1['parameters']
bn1 = param1['bn_running_stats']
现在已知：
1. parameters1里面每一个都是62长度的列表 的确包含bn的参数-weights bias 这些参数的确是相同的
2. bn1中每一个都是一个20长度的列表 对应于一个模型20个bn层 每一层里面又含有两个值 分别是running average  running variance 
这个bn1中的每个模型的值都互相不同 
现在推测：
这些不同的bn应该就对应了不同的fc层 即一个fc状态对应了一种bn层状态
"""