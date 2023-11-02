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
from main_mnist import get_model
from models import MNIST_277, MNIST_5066, Lenet, MNIST_466, MNIST_1066, MNIST_9914, MNIST_13354, MNIST_19664, MNIST_25974
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

def load_parameters(net, param):
    layer_idx = 0
    for p in net.parameters(): 
        p.data = param[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='traing epochs')
    parser.add_argument('--save', default='yes', type=str, help='whether save trained parameters')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_interval', default=10, type=int, help='save interval')
    parser.add_argument('--num_experts', default=100, type=int, help='number of models')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=5e-3, help='l2 regularization')
    parser.add_argument('--param_num', type=int, default=277, help='which size of model')

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

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

    ''' Train teacher model '''
    save_path, net = get_model(args)
    os.makedirs(save_path, exist_ok=True)

    num_params = sum([np.prod(p.size()) for p in (net.parameters())])
    print(f'parameters number is {num_params}')
    noisy_accs = {}
    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0 ]:
        noisy_accs[alpha] = 0

    clean_accs = []

    path = '../parameters/mnist_'+ str(args.param_num) + '/'
    params_files = os.listdir(path)
    total = 0
    for file in params_files[:1]:
        params = torch.load(path+file)
        for param in params:
            net = load_parameters(net, param)
            acc, loss = test(net.cuda(), testloader)
            print(f'=====Normal accuracy is {acc}=====')

            clean_accs.append(acc)
            tmp = param
            i = 0
            total += 1
            for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0 ]:
                ## add noise
                print(f'<<<<<<<<<<<<<<<<<<<<<<<The alpha is {alpha}>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                noise_std = 1.0
                _, net = get_model(args)
                net = load_parameters(net, tmp).cuda()

                for param in net.parameters():
                    if param.requires_grad:
                    # print(param.shape)
                    # if param.numel() == 2400:
                        noise = torch.randn(param.size(), device = device) * noise_std
                        param.data =  (1-alpha) * param.data + alpha * noise
                noisy_acc, loss = test(net.cuda(), testloader)
                noisy_accs[alpha] += noisy_acc
                i += 1
    # print(len(noisy_accs))
    # print(len(noisy_accs[0]))
    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0 ]:
        noisy_accs[alpha] /= total

    print(f'parameters number is {num_params}')
    print('Final noisy accs is', noisy_accs)
    print('Natural accs is', np.mean(np.array(clean_accs)))

                