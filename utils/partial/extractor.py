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
from models_cifar import ResNet18_cifar10_feature, classifier
from models_cifar100 import *
def load_params(params, model):
    ## params是一个列表形式存储的model.parameters()
    layer_idx = 0
    for p in model.parameters(): 
        p.data = params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            _,output = model(data)
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

def test_fc(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = torch.squeeze(model(data))
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


def extractor(model, train_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    features = []
    labels = []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            # print(feature.shape) 【256,512】

            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            feature_split = torch.split(feature, 1, dim=0)
            feature_list = [t for t in feature_split]
            features += feature_list
            label_split = torch.split(target, 1, dim=0)
            label_list = [t for t in label_split]
            labels += label_list
    test_loss /= total
    acc = 100. * correct / total
    print(f'extractor acc is {acc}')
    print('len of fea', len(features))
    return features, labels

def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)


def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)  # 这里使用Xavier初始化
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)
    return layer

def get_dataset(dataset):
    if dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('../../../datasets', train=True, download=True,
                            transform=transforms.Compose(
                                [
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])),
            batch_size=256, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('../../../datasets', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=256, shuffle=True, num_workers=4)
        return train_loader, test_loader
    elif dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('../../../datasets', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
        batch_size=512, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('../../../datasets', train=False, transform=transforms.Compose([
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
    parser.add_argument('--epochs', default=90, type=int, help='traing epochs')
    parser.add_argument('--save', default='yes', type=str, help='whether save trained parameters')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_interval', default=100, type=int, help='save interval')
    parser.add_argument('--num_experts', default=100, type=int, help='number of models')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=5e-3, help='l2 regularization')
    parser.add_argument('--param_num', type=str, default='res18', help='which size of model')
    parser.add_argument('--schedule',default=[40,75],help='when to change lr')
    parser.add_argument('--dataset',default='cifar')
    parser.add_argument('--extract',default='yes')
    parser.add_argument('--train',default='yes')



    args = parser.parse_args()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data
    print('==> Preparing data..')
    trainloader, testloader = get_dataset(args.dataset)

 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'cifar':
        num_class = 10
        if args.param_num == 'res18':
            net = ResNet18_cifar10_feature().cuda()
            trained_params = torch.load('../parameters/cifar_res18/parameters_aaa0612e-2.pt')[0]
            well_net = load_params(trained_params, net)
            save_path = '../features/cifar_res18'

        elif args.param_num == 'res50':
            net = ResNet50_cifar10_feature().cuda()
            trained_params = torch.load('../parameters/cifar_res50/parameters_9cbdb000-2.pt')[0]
            well_net = load_params(trained_params, net)
            save_path = '../features/cifar_res50'
        else:
            assert(0)
    elif args.dataset == 'cifar100':
        num_class=100
        if args.param_num == 'res18':
            net = ResNet18_cifar100_feature().cuda()
            trained_params = torch.load('../parameters/cifar100_res18/parameters_4c1f6bd0-2.pt')[0]
            well_net = load_params(trained_params, net)
            save_path = '../features/cifar100_res18'

        elif args.param_num == 'res50':
            net = ResNet50_cifar100_feature().cuda()
            trained_params = torch.load('../parameters/cifar100_res50/parameters_78d38c6c-2.pt')[0]
            well_net = load_params(trained_params, net)
            save_path = '../features/cifar100_res50'
        else:
            assert(0)

    os.makedirs(save_path, exist_ok=True)
    num_params = sum([np.prod(p.size()) for p in (net.parameters())])
    print(f'parameters number is {num_params}')
    well_net.eval()


    #提取训练集特征
    if args.extract == 'yes':
        print('========extract features========')
        data = trainloader if args.train == 'yes' else testloader
        features, labels = extractor(well_net.cuda(), data)
        #测试测试集正常性能
        print('========test train acc========')
        acc, loss = test(well_net.cuda(), trainloader)
        print('========test test acc========')
        acc, loss = test(well_net.cuda(), testloader)
        print("Saving {}".format(os.path.join(save_path, "features.pt")))
        name = 'train' if args.train == 'yes' else 'test'
        torch.save({'features':features,'labels':labels}, os.path.join(save_path, f"features_{name}.pt"))

    ## test for features
    if args.param_num == 'res18':
        fc = classifier_res18(num_classes=num_class).cuda()
    elif args.param_num == 'res50':
        fc = classifier_res50(num_classes=num_class).cuda()

    featuredata = torch.load(os.path.join(save_path, f"features_{name}.pt"))['features']
    featuredata = torch.stack(featuredata)
    featurelabel = torch.load(os.path.join(save_path, f"features_{name}.pt"))['labels']
    featurelabel = torch.tensor(featurelabel)
    featureset = torch.utils.data.TensorDataset(featuredata,featurelabel)
    featureloader = torch.utils.data.DataLoader(featureset,batch_size=255, shuffle=True, num_workers=0)
    
    well_fc = load_params(trained_params[-2:], fc)
    print(f'===========test well fc============')
    acc, loss = test_fc(well_fc.cuda(), featureloader)

