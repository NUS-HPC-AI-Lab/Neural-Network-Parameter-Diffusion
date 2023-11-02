import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('../lib')
from models_mnist import * 
from models_cifar import *
def get_net(param_num, dataset):
    if dataset == 'mnist':
        if param_num == '277':
            net = MNIST_277().cuda()
            save_path = '../parameters/mnist_277'
        elif param_num == '466':
            net = MNIST_466().cuda()
            save_path = '../parameters/mnist_466'
        elif param_num == '5066':
            net = MNIST_5066().cuda()
            save_path = '../parameters/mnist_5066'
        elif param_num == '9914':
            net = MNIST_9914().cuda()
            save_path = '../parameters/mnist_9914'
        elif param_num == '13354':
            net = MNIST_13354().cuda()
            save_path = '../parameters/mnist_13354'
        elif param_num == '19664':
            net = MNIST_19664().cuda()
            save_path = '../parameters/mnist_19664'
        elif param_num == '25974':
            net = MNIST_25974().cuda()
            save_path = '../parameters/mnist_25974'
        elif param_num == 'conv3':
            net = ConvNet(net_depth=3).cuda()
            save_path = '../parameters/mnist_conv3'
        elif param_num == 'conv4':
            net = ConvNet(net_depth=4).cuda()
            save_path = '../parameters/mnist_conv4'
        elif param_num == 'conv5':
            net = ConvNet(net_depth=5).cuda()
            save_path = '../parameters/mnist_conv5'
        else:
            assert(0)
        return save_path, net
    if dataset == 'cifar10':
        if param_num == '6766':
            net = cifar_6766().cuda()
            save_path = '../parameters/cifar_6766'
        elif param_num == '502':
            net = cifar_502().cuda()
            save_path = '../parameters/cifar_502'
        elif param_num == '405':
            net = cifar_405().cuda()
            save_path = '../parameters/cifar_405'
        elif param_num == '1138':
            net = cifar_1138().cuda()
            save_path = '../parameters/cifar_1138'
        elif param_num == '13314':
            net = cifar_13314().cuda()
            save_path = '../parameters/cifar_13314'
        elif param_num == 'conv3':
            net = ConvNet_cifar10(net_depth=3).cuda()
            save_path = '../parameters/cifar_conv3'
        elif param_num == 'res18':
            net = ResNet18_cifar10().cuda()
            save_path = '../parameters/cifar_res18'
        # elif param_num == 'conv4':
        #     net = ConvNet_cifar10(net_depth=4).cuda()
        #     save_path = '../parameters/cifar_13314'
        # elif param_num == 'conv5':
        #     net = ConvNet_cifar10(net_depth=5).cuda()
        #     save_path = '../parameters/cifar_13314'
        return save_path, net

def get_dataset(dataset):
    if dataset == 'mnist':
        print('==> Preparing mnist data..')

        transform = transforms.Compose([transforms.ToTensor(),])
                                    #    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

        trainset = torchvision.datasets.MNIST(root = "../../../datasets/",
                                    transform=transform,
                                    train = True,
                                    download = True)

        testset = torchvision.datasets.MNIST(root="../../../datasets/",
                                transform = transform,
                                train = False,
                                download = True)
        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=500, shuffle=True, num_workers=0)

        testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)
        return trainloader, testloader
    
    elif dataset == 'cifar10':
        print('==> Preparing cifar10 data..')
            
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../datasets', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
        batch_size=256, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=256, shuffle=False, num_workers=4)
        return train_loader, test_loader

def reverse_tomodel(flattened, model):
    example_parameters = [p for p in model.parameters()]
    len = 0
    reversed_params = []
    # layer = 0
    # import pdb; pdb.set_trace()
    for p in example_parameters:
        # if p.numel()==30720:
        #     # reversed_params.append(self.layer4[random.randint(0,9)].reshape(p.shape))
        #     reversed_params.append(self.layer4[0].reshape(p.shape))
        #     continue
        flattened_params = flattened[len: len+p.numel()]
        reversed_params.append(flattened_params.reshape(p.shape))
        len += p.numel()
        # layer += 1

            # reversed_params is saved as [p for p in model.parameters()]
    layer_idx = 0
    for p in model.parameters(): 
        p.data = reversed_params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def load_param(param_list, model):
    layer_idx = 0
    for p in model.parameters(): 
        p.data = param_list[layer_idx]
        p.data.to(torch.device('cuda'))
        # p.data.requires_grad_(True)
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

def test_em(model_list, test_loader):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            outputs = []
            for model in model_list:
                model.eval()
                model = model.cuda()
                output = model(data)
                outputs.append(output)
            stacked = torch.stack(outputs)
            mean_output = torch.mean(stacked, dim=0)

            target=target.to(torch.int64)
            test_loss += F.cross_entropy(mean_output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
        .format(test_loss, acc))
    return acc, test_loss

if __name__ == '__main__':
    param = torch.load("/home/pangshengyuan/code/padiff/parameters/cifar_res18/parameters_6673104a-2.pt")[0]
    import pdb;pdb.set_trace()
    _,net = get_net('res18','cifar10')
    net = load_param(param, net)
    _,testloader = get_dataset('cifar10')
    acc,_ = test(net.cuda(), testloader)


