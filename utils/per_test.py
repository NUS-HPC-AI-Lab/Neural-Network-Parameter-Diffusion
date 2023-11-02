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
sys.path.append('../data_utils/')
from tools import get_net, get_dataset
import sys
sys.path.append('../lib')
from models_mnist import * 
from models_cifar import *
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
    # print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
    #     .format(test_loss, acc))
    return acc, test_loss


def load_parameters(net, param):
    layer_idx = 0
    # print(len(param))
    for p in net.parameters(): 
        p.data = param[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--param_num', type=str, default=5066, help='which size of model')

    args = parser.parse_args()



    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data
    print('==> Preparing data..')
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.dataset)
    _, testloader = get_dataset(args.dataset)

    ''' Train teacher model '''
    save_path, net = get_net(str(args.param_num), args.dataset)
    os.makedirs(save_path, exist_ok=True)

    num_params = sum([np.prod(p.size()) for p in (net.parameters())])
    print(f'parameters number is {num_params}')
    noisy_accs = {}
    improve = {}
    best_noisy_accs = {}

    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]:
        noisy_accs[alpha] = 0
        improve[alpha] = 0
        best_noisy_accs[alpha] = 0


    clean_accs = []

    
    params_files = os.listdir(save_path)
    total = 0
    test_number = 1000
    file = params_files[0]
    params = torch.load(save_path+'/'+file)
    param_ori = params[0]
    for test_id in range(test_number):
                net = load_parameters(net, param_ori)
                acc, loss = test(net.cuda(), testloader)
                # print(f'=====Normal accuracy is {acc}=====')

                clean_accs.append(acc)
                tmp = param_ori
                i = 0
                # total += 1
                for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]:
                    ## add noise
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<The alpha is {alpha}>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    noise_std = 1.0
                    _, net = get_net(str(args.param_num), args.dataset)
                    net = load_parameters(net, tmp).cuda()

                    for param in net.parameters():
                        if param.requires_grad:
                        # print(param.shape)
                        # if param.numel() == 2400:
                            noise = torch.randn(param.size(), device = device) * noise_std
                            param.data =  (1-alpha) * param.data + alpha * noise
                    noisy_acc, loss = test(net.cuda(), testloader)
                    if noisy_acc > best_noisy_accs[alpha]:
                         best_noisy_accs[alpha] = noisy_acc
                    if noisy_acc > acc:
                        improve[alpha] += 1
                    noisy_accs[alpha] += noisy_acc
                    i += 1
    # print(len(noisy_accs))
    # print(len(noisy_accs[0]))
    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]:
        noisy_accs[alpha] /= test_number
        improve[alpha] /= test_number


    print(f'parameters number is {num_params}')
    print('Final noisy accs is', noisy_accs)
    print('improve accs ratio is', improve)
    print('best noisy accs is', best_noisy_accs)

    print('Natural accs is', np.mean(np.array(clean_accs)))

                