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
from torch.utils.data import Dataset

sys.path.append('../../lib/')
from models_imagenet import ConvNet_200
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

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    
def get_dataset(data_path='/home/pangshengyuan/datasets'):
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=0)
        return trainloader, testloader

# def get_dataset(data_path='../../../../datasets/tiny-imagenet-200'):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])

#     testdataset =torchvision.datasets.ImageFolder(root=data_path + '/val/',transform=transform_test)
#     traindataset =torchvision.datasets.ImageFolder(root=data_path + '/train/',transform=transform_test)
#     trainloader = torch.utils.data.DataLoader(traindataset, batch_size=256, shuffle=True, num_workers=0)
#     testloader = torch.utils.data.DataLoader(testdataset, batch_size=256, shuffle=False, num_workers=0)

    
#     return trainloader, testloader



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  #imagenet conv3是0.01
    parser.add_argument('--epochs', default=75, type=int, help='traing epochs')
    parser.add_argument('--save', default='yes', type=str, help='whether save trained parameters')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_interval', default=10, type=int, help='save interval')
    parser.add_argument('--num_experts', default=100, type=int, help='number of models')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=5e-3, help='l2 regularization')
    parser.add_argument('--param_num', type=str, default='vit', help='which size of model')
    parser.add_argument('--schedule',default=[50,75],help='when to change lr')

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
        if args.param_num == 'conv3':
            net = ConvNet_200(net_depth=3).cuda()
            threshold = 20
            save_path = '../../parameters/imagenet_conv3'
        elif args.param_num == 'conv4':
            net = ConvNet_200(net_depth=4).cuda()
            threshold = 35
            save_path = '../../parameters/imagenet_conv4'
        elif args.param_num == 'conv5':
            net = ConvNet_200(net_depth=5).cuda()
            threshold = 30
            save_path = '../../parameters/imagenet_conv5'
        elif args.param_num == 'vit':
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
            threshold = 60
            save_path = '../../parameters/imagenet_vit'

        else:
            assert(0)
        os.makedirs(save_path, exist_ok=True)
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
            parameters = []
            # break