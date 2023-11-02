"""
本文件做bn层的finetune  仅支持ers18 cifar10
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
sys.path.append('../lib/')
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

def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)

def initialize_bn_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)
    return model

def save_bn(model):
    bn_param = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
                bn_param.append(module.weight.data.clone())
                bn_param.append(module.bias.data.clone())

    return bn_param

def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)  # 这里使用Xavier初始化
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)
    return layer

def get_dataset():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../../datasets', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
        batch_size=512, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=256, shuffle=True, num_workers=4)
    return train_loader, test_loader

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='traing epochs')
    parser.add_argument('--save', default='yes', type=str, help='whether save trained parameters')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_interval', default=100, type=int, help='save interval')
    parser.add_argument('--num_experts', default=150, type=int, help='number of models')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=5e-3, help='l2 regularization')
    parser.add_argument('--param_num', type=str, default='res18', help='which size of model')
    parser.add_argument('--schedule',default=[40,75],help='when to change lr')

    args = parser.parse_args()



    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data
    print('==> Preparing data..')

    trainloader, testloader = get_dataset()

    parameters = []

    for it in range(0, args.num_experts):
        seed = int(time.time())
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)   
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
        if args.param_num == 'res18':
            net = ResNet18_cifar10().cuda()
            trained_params = torch.load('../parameters/cifar_res18/parameters_aaa0612e-2.pt')[0]
            well_net = load_params(trained_params, net)
            save_path = '../parameters_onlybn/cifar_res18'
            bn_params = []
            for name,module in well_net.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_params.append(module.weight.data.clone())
                    bn_params.append(module.bias.data.clone())
            # print(bn_params)
            
        # elif args.param_num == 'conv3':
        #     net = ConvNet_cifar10(net_depth=3).cuda()
        #     trained_params = torch.load('../parameters/cifar_conv3_new/parameters_2ce448d6-2.pt')[0]
        #     well_net = load_params(trained_params, net)
        #     save_path = '../parameters_onlyfc/cifar_conv3_new'
        # elif args.param_num == 'conv4':
        #     net = ConvNet_cifar10(net_depth=4).cuda()
        #     trained_params = torch.load('')[0]
        #     well_net = load_params(trained_params, net)
        #     save_path = '../parameters_onlyconv/cifar_conv4'       
        # elif args.param_num == 'conv5':
        #     net = ConvNet_cifar10(net_depth=5).cuda()
        #     trained_params = torch.load('')[0]
        #     well_net = load_params(trained_params, net)
        #     save_path = '../parameters_onlyconv/cifar_conv5'

        else:
            assert(0)
        os.makedirs(save_path, exist_ok=True)
        num_params = sum([np.prod(p.size()) for p in (net.parameters())])
        print(f'parameters number is {num_params}')
        well_net.train()

        well_net = initialize_bn_parameters(well_net)
        for name, module in well_net.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        for name, param in well_net.named_parameters():
            print(name, param.requires_grad)
        # pdb.set_trace()
        optimizer = torch.optim.SGD(well_net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.l2) 
        optimizer.zero_grad()
        threshold = -1

        for e in range(args.epochs):
            print(f"====================Number{it} model \t Epoch:{e}====================")
            train(well_net.cuda(), trainloader, optimizer)
            adjust_learning_rate(optimizer, e, args)
            acc, loss = test(well_net.cuda(), testloader)
            if e > 30:
                if acc > threshold:
                    #仅存储bn层参数
                    parameters.append(save_bn(well_net))
       
                    if len(parameters) == args.save_interval:  #10 teacher models saved to one buffer.pt
                        
                        # get uuid
                        n = str(uuid.uuid1())[:10]
                        while os.path.exists(os.path.join(save_path, "parameters_{}.pt".format(n))):
                            n = str(uuid.uuid1())[:10]
                        print("Saving {}".format(os.path.join(save_path, "parameters_{}.pt".format(n))))
                        torch.save(parameters, os.path.join(save_path, "parameters_{}.pt".format(n)))
                        parameters = []
            # break