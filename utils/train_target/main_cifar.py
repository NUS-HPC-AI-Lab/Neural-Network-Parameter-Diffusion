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
from pytorch_lightning import seed_everything

sys.path.append('../lib/')
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
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            # if idx==0:
            #     print('output',output.shape, output)
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
        batch_size=256, shuffle=True, num_workers=4)
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

def load_param(param_list, model):
    layer_idx = 0
    for p in model.parameters(): 
        p.data = param_list[layer_idx]
        p.data.to(torch.device('cuda'))
        # p.data.requires_grad_(True)
        layer_idx += 1
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='traing epochs')
    parser.add_argument('--save', default='yes', type=str, help='whether save trained parameters')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_interval', default=1, type=int, help='save interval')
    parser.add_argument('--num_experts', default=1, type=int, help='number of models')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularization')
    parser.add_argument('--param_num', type=str, default='res18', help='which size of model')
    parser.add_argument('--schedule',default=[40,75,90,100],help='when to change lr')

    args = parser.parse_args()



    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data
    print('==> Preparing data..')

    trainloader, testloader = get_dataset()

    parameters = []
    bn_running_stats = []
    for it in range(0, args.num_experts):
        seed = int(time.time())
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)   
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.param_num == '405':
            net = cifar_405().cuda()
            threshold = 20
            save_path = '../parameters/cifar_405'
        elif args.param_num == '1138':
            net = cifar_1138().cuda()
            threshold = 35
            save_path = '../parameters/cifar_1138'
        elif args.param_num == '502':
            net = cifar_502().cuda()
            threshold = 30
            save_path = '../parameters/cifar_502'
        elif args.param_num == '6766':
            net = cifar_6766().cuda()
            threshold = 45
            save_path = '../parameters/cifar_6766'
        elif args.param_num == 9914:
            net = cifar_9914().cuda()
            save_path = '../parameters/cifar_9914'
        elif args.param_num == 13354:
            net = cifar_13354().cuda()
            save_path = '../parameters/cifar_13354'
        elif args.param_num == 19664:
            net = cifar_19664().cuda()
            save_path = '../parameters/cifar_19664'
        elif args.param_num == 25974:
            net = cifar_25974().cuda()
            save_path = '../parameters/cifar_25974'
        elif args.param_num == 'conv3':
            net = ConvNet_cifar10(net_depth=3).cuda()
            save_path = '../parameters/cifar_conv3_new'
        elif args.param_num == 'conv4':
            net = ConvNet_cifar10(net_depth=4).cuda()
            save_path = '../parameters/cifar_conv4'        
        elif args.param_num == 'conv5':
            net = ConvNet_cifar10(net_depth=5).cuda()
            save_path = '../parameters/cifar_conv5'
        elif args.param_num == 'res18':
            net = ResNet18_cifar10().cuda()
            save_path = '../parameters/cifar_res18'
        elif args.param_num == 'res50':
            net = ResNet50_cifar10().cuda()  #23520842
            save_path = '../parameters/cifar_res50'
        else:
            assert(0)
        os.makedirs(save_path, exist_ok=True)
        num_params = sum([np.prod(p.size()) for p in (net.parameters())])
        print(f'parameters number is {num_params}')
        net = net.to(device)
        net.train()
        # pdb.set_trace()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.l2) 
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 

        optimizer.zero_grad()
        threshold = -1

        for e in range(args.epochs):
            print(f"====================Number{it} model \t Epoch:{e}====================")
            train(net, trainloader, optimizer)
            adjust_learning_rate(optimizer, e, args)
            acc, loss = test(net, testloader)
        
        if acc > threshold:
            parameters.append([p.detach().cpu() for p in net.parameters()])
     
        else: 
            pass 
        if len(parameters) == args.save_interval:  #10 teacher models saved to one buffer.pt
            
            # get uuid
            n = str(uuid.uuid1())[:10]
            while os.path.exists(os.path.join(save_path, "parameters_{}.pt".format(n))):
                n = str(uuid.uuid1())[:10]
            print("Saving {}".format(os.path.join(save_path, "parameters_{}.pt".format(n))))
            torch.save(parameters, os.path.join(save_path, "parameters_{}.pt".format(n)))
            # import pdb; pdb.set_trace()

            parameters = []
            bn_running_stats = []
            # break