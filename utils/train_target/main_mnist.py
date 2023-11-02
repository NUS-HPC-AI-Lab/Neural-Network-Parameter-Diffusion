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
import timm
import pdb
import time
import uuid
import sys
sys.path.append('../lib/')
from models_mnist import *

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
    parser.add_argument('--param_num', type=str, default='277', help='which size of model')

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





    parameters = []

    for it in range(0, args.num_experts):
        seed = int(time.time())
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)   
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=500, shuffle=True, num_workers=0, worker_init_fn=_init_fn)

        testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

        ''' Train teacher model '''
        if args.param_num == '277':
            net = MNIST_277().cuda()
            save_path = '../parameters/mnist_277'
        elif args.param_num == '466':
            net = MNIST_466().cuda()
            save_path = '../parameters/mnist_466'
        elif args.param_num == '5066':
            net = MNIST_5066().cuda()
            save_path = '../parameters/mnist_5066'
        elif args.param_num == '9914':
            net = MNIST_9914().cuda()
            save_path = '../parameters/mnist_9914'
        elif args.param_num == '13354':
            net = MNIST_13354().cuda()
            save_path = '../parameters/mnist_13354'
        elif args.param_num == '19664':
            net = MNIST_19664().cuda()
            save_path = '../parameters/mnist_19664'
        elif args.param_num == '25974':
            net = MNIST_25974().cuda()
            save_path = '../parameters/mnist_25974'
        elif args.param_num == 'conv3':
            net = ConvNet(net_depth=3).cuda()
            save_path = '../parameters/mnist_conv3'
        elif args.param_num == 'conv5':
            net = ConvNet(net_depth=5).cuda()
            save_path = '../parameters/mnist_conv5'
        elif args.param_num == 'conv4':
            net = ConvNet(net_depth=4).cuda() #450186 
            save_path = '../parameters/mnist_conv4'
        elif args.param_num == 'conv1_1stride':
            net = ConvNet_1stride(net_depth=1).cuda() #一百二十万
            save_path = '../parameters/mnist_conv1stride1'
        elif args.param_num == 'conv1_nopool':
            net = ConvNet(net_depth=1, net_pooling='none').cuda() #一百三十万 conv3一百六十万
            save_path = '../parameters/mnist_conv1nopool'
        # elif args.param_num == 'conv5_nopool':
        #     net = ConvNet(net_depth=5,net_pooling='none').cuda() # 
        #     save_path = '../parameters/mnist_conv5nopool'
        elif args.param_num == 'lenet':
            net = Lenet().cuda()  #lenet就是44426
            save_path = '../parameters/mnist_conv5'
        elif args.param_num == 'resnet18':
            net = ResNet18().cuda()    #11172810
            save_path = '../parameters/mnist_resnet18'
        elif args.param_num == 'resnet10':
            net = ResNet10().cuda()    #4902090   vgg11是九百万
            save_path = '../parameters/mnist_resnet10'
        elif args.param_num == 'vit':
            net = timm.create_model('vit_tiny_patch16_224',fe)
        else:
            assert(0)
        os.makedirs(save_path, exist_ok=True)
        # net = Lenet()
        # params = [p.detach().cpu() for p in net.parameters()]
        # i=1
        # for param in params:
        #     print(f'Number {i} layer')
        #     print(f'parameter shape is:{param.shape}')
        #     print(f'parameter number:{param.numel()}')
        #     i += 1
        # assert(0)

        num_params = sum([np.prod(p.size()) for p in (net.parameters())])
        print(f'parameters number is {num_params}')
        net = net.to(device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.l2) 
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 

        optimizer.zero_grad()

        for e in range(args.epochs):
            print(f"====================Number{it} model \t Epoch:{e}====================")
            train(net, trainloader, optimizer)
            acc, loss = test(net, testloader)
        
        if acc > 85:
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
            parameters = []
            # break