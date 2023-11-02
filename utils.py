"""
本脚本中的partial相关函数中，只保留了res18相关，（之前conv系列的fc可以重新补充上来，应该只需要将target_layer设置为'classifier'即可，
不过需要注意的是，conv系列当时存储的都是全模型参数（不是部分目标参数）
"""

from lib.models_cifar import *
from lib.models_mnist import *
from lib.models_imagenet import *
from lib.models_cifar100 import *

import numpy as np
import os
import sys
import time
from torch.utils.data import Dataset
import timm
from dataset_folder import ImageFolder
from timm.models import create_model
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
# todo: res18 res50 res101  cifar10 cifar100
class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    

def get_testdata(dataset,network=None):
    # import pdb;pdb.set_trace()

    if dataset == 'mnist':
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root="../../datasets/",
                           transform = transforms.Compose([transforms.ToTensor(),]),
                           train = False, download=True), batch_size=100, shuffle=False, num_workers=0, )
    elif dataset == 'cifar':
            testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=256, shuffle=False, num_workers=4)
    elif dataset == 'cifar100':
            testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('./datasets', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])),
        batch_size=512, shuffle=False, num_workers=4)
    elif dataset == 'imagenet':
        net = create_model(
            network,
            pretrained=True,
            num_classes=1000,
        )
        data_config = timm.data.resolve_model_data_config(net)
        transforms_timm = timm.data.create_transform(**data_config, is_training=False)
        testloader = torch.utils.data.DataLoader(
            ImageFolder('/home/wangkai/zpxu/Imagenet/val', transforms_timm),
            batch_size=2048, shuffle=False,
            num_workers=0, pin_memory=True)

    return testloader
def get_evaldata(dataset,len_s=None,network=None):

    if dataset == 'mnist':
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root="../../datasets/",
                           transform = transforms.Compose([transforms.ToTensor(),]),
                           train = True, download=True), batch_size=100, shuffle=False, num_workers=0, )
    elif dataset == 'cifar':
            testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../datasets', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=256, shuffle=True, num_workers=4)
    elif dataset == 'cifar100':
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('./datasets', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
        batch_size=256, shuffle=False, num_workers=4)
    elif dataset == 'imagenet':
        net = create_model(
            network,
            pretrained=True,
            num_classes=1000,
        )
        data_config = timm.data.resolve_model_data_config(net)
        transforms_timm = timm.data.create_transform(**data_config, is_training=True)
        testloader = torch.utils.data.DataLoader(
            ImageFolder('/home/wangkai/zpxu/Imagenet/train', transforms_timm,len_s = len_s),
            batch_size=2048, shuffle=True,
            num_workers=0, pin_memory=True)

    return testloader
def get_traindata(dataset):
    if dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./datasets', train=True, download=True,
                            transform=transforms.Compose(
                                [
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])),
            batch_size=512, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=256, shuffle=True, num_workers=4)
        return train_loader, test_loader
    if dataset == 'cifar100':
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('./datasets', train=True, download=True, transform=transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])),
            batch_size=512, shuffle=True, num_workers=4)
        return trainloader

def load_params(params, model):
    ## params是一个列表形式存储的model.parameters()
    layer_idx = 0
    for p in model.parameters(): 
        p.data = params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def test_generated(self, param):
    # import pdb;pdb.set_trace()
    net = get_net(self.dataset, self.num_params_data)
    ##check for matching of model and fetched parameters
    target_num = sum([np.prod(p.size()) for p in (net.parameters())])
    params_num = torch.squeeze(param).shape[0] #+ 30720
    # print(f'target size is {target_num}, load param size is {params_num}')
    # import pdb;pdb.set_trace()
    assert(target_num==params_num)

    param = torch.squeeze(param) # [1,5066]--> [5066]
    # de_norm_param = de_normalize(param)
    net = reverse_tomodel(param, net).cuda()
    acc, loss = test(net, self.testloader)
    del net
    return acc


def test_generated_partial(self, param,dataloader,fea_path=None):
    # import pdb;pdb.set_trace()
    net = self.net
    target_num = 0
    if self.target_layer == 'bn':
        for name, module in net.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                if hasattr(module, 'weight'):
                    target_num += torch.numel(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    target_num += torch.numel(module.bias)
    elif self.target_layer == 'bn-2':
        tar_layers = ['layer4.1.bn1.weight','layer4.1.bn1.bias','layer4.1.bn2.weight','layer4.1.bn2.bias']
        for name, tmp_param in net.named_parameters():
                if name in tar_layers:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'conv_fc':
        for name, tmp_param in net.named_parameters():
                if ('layer4.1.conv2' in name) or ('linear' in name):
                    target_num += tmp_param.numel()
    elif self.target_layer == 'conv4_1_2':
        for name, tmp_param in net.named_parameters():
                if 'layer4.1.conv2' in name:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'r50_bn':
        for name, tmp_param in net.named_parameters():
                if name in ['layer4.2.bn3.weight','layer4.2.bn3.bias']:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'r18_bn':
        for name, tmp_param in net.named_parameters():
                if name in ['layer4.1.bn2.weight','layer4.1.bn2.bias']:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'vgg_fc':
        for name, tmp_param in net.named_parameters():
                if name in ['head.fc.weight','head.fc.bias']:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'tinyvit_norm':
        for name, tmp_param in net.named_parameters():
                if name in ['norm.weight','norm.bias']:
                    target_num += tmp_param.numel()
    elif self.target_layer == 'linear':  #linear for res18's fc
        for name, tmp_param in net.named_parameters():
            if self.target_layer in name:
                target_num += tmp_param.numel()
    params_num = torch.squeeze(param).shape[0] #+ 30720
    assert(target_num==params_num)

    param = torch.squeeze(param)
    net = partial_reverse_tomodel(self, param, net).cuda()
    acc, loss = test(net, dataloader,fea_path=fea_path)
    del net
    return acc

def test(model, test_loader,fea_path=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        if fea_path is not None:
            expert_files = os.listdir(fea_path)
            for m in expert_files:
                models = fea_path + m
                fea_targets = torch.load(models)
                targets = fea_targets[1].to('cuda')
                inputs = fea_targets[0].to('cuda')
                outputs = model.forward_norm(inputs)
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #results+=(predicted.eq(targets.view_as(predicted)).cpu().numpy().tolist())
            acc = 100.*correct/total
            return acc, _
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
    #       .format(test_loss, acc))
    del model
    return acc, test_loss

def reverse_tomodel(flattened, model):
    example_parameters = [p for p in model.parameters()]
    length = 0
    reversed_params = []

    for p in example_parameters:
        flattened_params = flattened[length: length+p.numel()]
        reversed_params.append(flattened_params.reshape(p.shape))
        length += p.numel()

    layer_idx = 0
    for p in model.parameters(): 
        p.data = reversed_params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def partial_reverse_tomodel(self,flattened, model):
    example_parameters = [p for p in model.parameters()]
    length = 0
    reversed_params = []
    if self.target_layer == 'linear':
        for p in example_parameters[-2:]:
            flattened_params = flattened[length: length+p.numel()]
            reversed_params.append(flattened_params.reshape(p.shape))
            length += p.numel()
            layer_idx = 0
        for pa in list(model.parameters())[-2:]: 
            pa.data = reversed_params[layer_idx]
            pa.data.to(torch.device('cuda'))
            layer_idx += 1

    elif self.target_layer == 'bn':
        length = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                p = module.weight.data
                flattened_params = flattened[length: length+p.numel()]
                length = length+p.numel()
                module.weight.data = flattened_params.reshape(p.shape)

                p = module.bias.data
                flattened_params = flattened[length: length+p.numel()]
                length = length+p.numel()
                module.bias.data = flattened_params.reshape(p.shape)
    elif self.target_layer == 'bn-2':
        tar_layers = ['layer4.1.bn1.weight','layer4.1.bn1.bias','layer4.1.bn2.weight','layer4.1.bn2.bias']
        layer_idx = 0
        for name,pa in model.named_parameters():
            if name in tar_layers:
                pa.data = flattened[layer_idx:layer_idx+pa.shape[0]].reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
                layer_idx += pa.shape[0]
    elif self.target_layer == 'conv4_1_2':
        for name, pa in model.named_parameters():
            if 'layer4.1.conv2' in name:
                pa.data = flattened.reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
    elif self.target_layer == 'r50_bn':
        tar_layers = ['layer4.2.bn3.weight','layer4.2.bn3.bias']
        layer_idx = 0
        for name,pa in model.named_parameters():
            if name in tar_layers:
                pa.data = flattened[layer_idx:layer_idx+pa.shape[0]].reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
                layer_idx += pa.shape[0]
    elif self.target_layer == 'tinyvit_norm':
        tar_layers = ['norm.weight','norm.bias']
        layer_idx = 0
        for name,pa in model.named_parameters():
            if name in tar_layers:
                pa.data = flattened[layer_idx:layer_idx+pa.shape[0]].reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
                layer_idx += pa.shape[0]
    elif self.target_layer == 'r18_bn':
        tar_layers = ['layer4.1.bn2.weight','layer4.1.bn2.bias']
        layer_idx = 0
        for name,pa in model.named_parameters():
            if name in tar_layers:
                pa.data = flattened[layer_idx:layer_idx+pa.shape[0]].reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
                layer_idx += pa.shape[0]
    elif self.target_layer == 'vgg_fc':
        tar_layers = ['head.fc.weigh','head.fc.bias']
        layer_idx = 0
        for name,pa in model.named_parameters():
            if name in tar_layers:
                pa.data = flattened[layer_idx:layer_idx+pa.shape[0]].reshape(pa.shape)
                pa.data.to(torch.device('cuda'))
                layer_idx += pa.shape[0]
    elif self.target_layer == 'conv_fc':
        length = 0   
        example_parameters = [example_parameters[-5],example_parameters[-2],example_parameters[-1]]
        length = 0
        for p in example_parameters:
            flattened_params = flattened[length: length+p.numel()]
            reversed_params.append(flattened_params.reshape(p.shape))
            length += p.numel()
        layer_idx = 0
        for name, pa in model.named_parameters():
                if ('layer4.1.conv2' in name) or ('linear' in name):      
                        pa.data = reversed_params[layer_idx]
                        pa.data.to(torch.device('cuda'))
                        layer_idx += 1
    
    return model

def save_best10(self, accs, params):  #params是一维向量
        sorted_list = sorted(accs, reverse=True)[:10]
        max_indices = [accs.index(element) for element in sorted_list]
        best_params = params[max_indices,:]
        del params
        return best_params

def save_best1(self, accs, params):  #params是一维向量
        sorted_list = sorted(accs, reverse=True)[:1]
        max_indices = [accs.index(element) for element in sorted_list]
        best_params = params[max_indices,:]
        del params
        return best_params
def top_acc_params(self, accs, params,topk):  #params是一维向量
        sorted_list = sorted(accs, reverse=True)[:topk]
        max_indices = [accs.index(element) for element in sorted_list]
        best_params = params[max_indices,:]
        del params
        return best_params

def test_ensem(self, best_params, net):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    ensemble_model = reverse_tomodel(mean, net)
    acc,_= test(ensemble_model.cuda(), self.testloader)
    del best_params
    return acc

def test_ensem_partial(self, best_params,dataloader,fea_path=None):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    acc = test_generated_partial(self, mean,dataloader,fea_path=fea_path)
    del best_params
    return acc

def test_ensem_inference(self, best_params, net):
    stacked = torch.stack(best_params)
    mean = torch.mean(stacked, dim = 0)
    ensemble_model = reverse_tomodel(torch.squeeze(mean), net)
    acc,_= test(ensemble_model.cuda(), self.testloader)
    del best_params
    return acc


def get_net(dataset, param_num,network=None):
    # import pdb;pdb.set_trace()

    if dataset == 'cifar':

        #因为本函数是 total diffusion  partial diffusion共用的 所以要先将
        #partial number逆回去
        if param_num == 20490:
            param_num = 'conv3'
            #param_num = 'res50'  #conv3 res50的特征维度相同

        elif param_num == 5130:  #res18的fc
            param_num = 'res18'
        elif param_num == 9600: #res18的bn
            param_num = 'res18'
        elif param_num == 2364426:
            param_num = 'res18'
        else:
            pass
        #############################################################
        if param_num == 405:
            net = cifar_405().cuda()
        elif param_num == 1138:
            net = cifar_1138().cuda()
        elif param_num == 502:
            net = cifar_502().cuda()
        elif param_num == 6766:
            net = cifar_6766().cuda()
        elif param_num == 13314:
            net = cifar_13314().cuda()
        elif param_num == 'conv3':
            net = ConvNet_cifar10(net_depth=3).cuda()
        elif param_num == 'res18':
            net = ResNet18_cifar10().cuda()
        elif param_num == 'res50':
            net = ResNet50_cifar10().cuda()
            
        else:
            assert(0)
        return net
    elif dataset == 'mnist':
        if param_num == 277:
            net = MNIST_277().cuda()
        elif param_num == 466:
            net = MNIST_466().cuda()
        elif param_num == 1066:
            net = MNIST_1066().cuda()
        elif param_num == 5066:
            net = MNIST_5066().cuda()
        elif param_num == 9914:
            net = MNIST_9914().cuda()
        elif param_num == 13354:
            net = MNIST_13354().cuda()
        elif param_num == 19664:
            net = MNIST_19664().cuda()
        elif param_num == 25974:
            net = MNIST_25974().cuda()
        elif param_num == 317706:### convnet3
            net = ConvNet(net_depth=3).cuda()
        elif param_num == 450186:### convnet3
            net = ConvNet(net_depth=4).cuda()
        elif param_num == 44426:### lenet
            net = Lenet().cuda()
        elif param_num == 594186:### convnet5
            net = ConvNet(net_depth=5).cuda()
        else:
            assert(0)
    elif dataset == 'cifar100':
        #### reverse#
        if param_num == 51300:
            param_num = 'res18'
        elif param_num == 204900:
            param_num = 'res50'
        elif param_num == 2048:
            param_path = get_good(dataset, param_num)
            param_num = 'res18'
        elif param_num == 2359296:
            param_path = get_good(dataset, param_num)
            param_num = 'res18'
        #############
        if param_num == 504420:  ##conv3
            net = ConvNet_100(net_depth=3).cuda()
        elif param_num == 'res18':
            net = ResNet18(100).cuda()
            checkpoint = torch.load(param_path)
            net.load_state_dict(checkpoint)
        elif param_num == 'res50':
            net = ResNet50_cifar100().cuda()
        else:
            assert(0)

    elif dataset == 'imagenet':
        if param_num == 857160:
            net = ConvNet_200(net_depth=4).cuda()
        elif param_num == 1938120:
            net = ConvNet_200(net_depth=3).cuda()
        elif param_num == 1938120:
            net = ConvNet_200(net_depth=3).cuda()
        elif network is not None:
            net = timm.create_model(network, pretrained=True).cuda()
        else: 
            assert(0)
    return net

##psy
def get_good(dataset, param_num):
    ###############reverse
    if dataset == 'cifar':
        if param_num == 20490:
            # param_num = 320010  #cifar10 conv3还有res50都是20490
            param_num = 'res50'
        elif param_num == 5130:
            param_num = 'res18'
        elif param_num == 9600:
            param_num = 'res18'
        elif param_num == 2364426:
            param_num = 'res18'
    if dataset == 'cifar100':
        if param_num == 204900:
            # param_num = 320010  #cifar10 conv3还有res50都是20490
            param_num = 'res50'
        elif param_num == 51300:
            param_num = 'res18'
        elif param_num == 2048:
            param_num = 'res18'
        elif param_num == 2359296:
            param_num = 'res18'
    ####################
    if dataset == 'cifar':
        if param_num == 320010:
            good_param_path = './parameters/cifar_conv3/parameters_8a3610c6-1.pt'
        elif param_num == 'res18':
            good_param_path = './parameters/cifar_res18/parameters_aaa0612e-2.pt'
        elif param_num == 'res50':
            good_param_path = './parameters/cifar_res50/parameters_9cbdb000-2.pt'
    elif dataset == 'cifar100':
        if param_num == 'res18':
            good_param_path = './parameters/ResNet18_cifar100/bn2full_model.pkl'
        elif param_num == 'res50':
            good_param_path = './parameters/cifar100_res50/parameters_78d38c6c-2.pt'
        
    return good_param_path
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f