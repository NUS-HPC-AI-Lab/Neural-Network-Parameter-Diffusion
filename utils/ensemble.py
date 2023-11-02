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
from tools import get_net, get_dataset, reverse_tomodel, load_param, test, test_em
sys.path.append('../lib/')
from models_cifar import *




def ensemble(net, params, type, testloader):
    ##输入列表存储格式的模型参数列表
    if type == 'param_en':
        if isinstance(params[0], list):
            # import pdb; pdb.set_trace()
            tmp = [torch.cat([p.reshape(-1) for p in param], 0) for param in params ]
        else:
            tmp = params  #一个列表 每个元素是一个一维参数向量
        stacked = torch.stack(tmp) #若取的是训练数据的模型 应去掉torch.squeeze和 list()
        mean = torch.mean(stacked, dim = 0)
        ensemble_model = reverse_tomodel(mean, net)
        test(ensemble_model.cuda(), testloader)

    elif type == 'pre_en':
        model_list = []

        if isinstance(params[0], list):
            for i in range(len(params)):
                model_list.append(load_param(params[i], net))
        else:
            for i in range(len(params)):
                model_list.append(reverse_tomodel(torch.squeeze(params[i]), net))
        test_em(model_list, testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--num', default=10, type=int, help='number of models to ensemble')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--param', default='conv3', type=str, help='param num')
    parser.add_argument('--type', default='param_en', type=str, help='ensemble object')

    args = parser.parse_args()

    trainloader, testloader = get_dataset(args.dataset)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data


    parameters = []

       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, net = get_net(args.param, args.dataset)
    save_path = './../parameters/cifar_conv3'
    files = os.listdir(save_path)
    params = []
    for file in files:
        parameters = torch.load(save_path+'/'+file) 
        for one in parameters:
            params.append(one)
    params = params[:4]
    print(len(params))
    num_params = sum([np.prod(p.size()) for p in (net.parameters())])
    print(f'parameters number is {num_params}')
    net = net.to(device)

    ensemble(net, params, args.type, testloader)
"""
这几个模型分别都是好的：92 93附近
参数融合会寄掉：前三个融合会降到34左右，前五个融合会降到10
logits融合acc是正常的，但是并不会相比于单个模型有提升
"""