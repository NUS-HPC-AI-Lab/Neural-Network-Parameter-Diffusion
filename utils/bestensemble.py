"""
本文件对partial fc进行ensemble
含有参数融合和logit融合
可以输入生成的fc或者good fc
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
from tools import get_net, get_dataset, reverse_tomodel, load_param, test, test_em
sys.path.append('../lib/')
from models_cifar import *
from models_cifar100 import *



def test_ensem_partial(self, best_params):
    #best_params是列表
    stacked = torch.stack(best_params)
    mean = torch.mean(stacked, dim = 0)
    acc = test_generated_partial(self, mean)
    del best_params
    return acc

def get_good(dataset, param_num):
    ###############reverse
    if dataset == 'cifar':
        if param_num == 20490:
            # param_num = 320010  #cifar10 conv3还有res50都是20490
            param_num = 'res50'
        elif param_num == 5130:
            param_num = 'res18'
    if dataset == 'cifar100':
        if param_num == 204900:
            # param_num = 320010  #cifar10 conv3还有res50都是20490
            param_num = 'res50'
        elif param_num == 51300:
            param_num = 'res18'
    ####################
    if dataset == 'cifar':
        if param_num == 320010:
            good_param_path = '../parameters/cifar_conv3/parameters_8a3610c6-1.pt'
        elif param_num == 'res18':
            good_param_path = '../parameters/cifar_res18/parameters_aaa0612e-2.pt'
        elif param_num == 'res50':
            good_param_path = '../parameters/cifar_res50/parameters_9cbdb000-2.pt'
    elif dataset == 'cifar100':
        if param_num == 'res18':
            good_param_path = '../parameters/cifar100_res18/parameters_4c1f6bd0-2.pt'
        elif param_num == 'res50':
            good_param_path = '../parameters/cifar100_res50/parameters_78d38c6c-2.pt'
        
    return good_param_path


def get_net(dataset, param_num):
    # import pdb;pdb.set_trace()

    if dataset == 'cifar':

        #因为本函数是 total diffusion  partial diffusion共用的 所以要先将
        #partial number逆回去
        if param_num == 20490:
            # param_num = 'conv3'
            param_num = 'res50'  #conv3 res50的特征维度相同

        elif param_num == 5130:
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
        #############
        if param_num == 504420:  ##conv3
            net = ConvNet_100(net_depth=3).cuda()
        elif param_num == 'res18':
            net = ResNet18_cifar100().cuda()
        elif param_num == 'res50':
            net = ResNet50_cifar100().cuda()
        else:
            assert(0)

    elif dataset == 'imagenet':
        if param_num == 857160:
            net = ConvNet_200(net_depth=4).cuda()
        elif param_num == 1938120:
            net = ConvNet_200(net_depth=3).cuda()
        else: 
            assert(0)
    return net

def test_generated_partial(self, param):
    # import pdb;pdb.set_trace()
    
    net = get_net(self.dataset, self.size)
    param_path = get_good(self.dataset, self.num_params_data)
    # import pdb;pdb.set_trace()

    ##check for matching of model and fetched parameters
    # print(self.target_layer)
    target_num = 0
    for name, tmp_param in net.named_parameters():
        # print(name)
        if self.target_layer in name:
             target_num += tmp_param.numel()
    params_num = torch.squeeze(param).shape[0] #+ 30720
    # print(f'target size is {target_num}, load param size is {params_num}')
    # import pdb;pdb.set_trace()
    assert(target_num==params_num)

    param = torch.squeeze(param) # [1,5066]--> [5066]
    # if self.num_params_data != 5130:
    good_params = torch.load(param_path)[0]
    well_model = load_params(good_params, net)
    net = partial_reverse_tomodel(param, well_model).cuda()
    acc, loss = test(net, testloader)
    del net
    return acc

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
                output = torch.squeeze(model(data))
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

def load_params(params, model):
    ## params是一个列表形式存储的model.parameters()
    example_parameters = [p for p in model.parameters()]
    length = 0
    reversed_params = []

    for p in example_parameters:
        flattened_params = params[length: length+p.numel()]
        reversed_params.append(flattened_params.reshape(p.shape))
        length += p.numel()
    layer_idx = 0
    for p in model.parameters(): 
        p.data = reversed_params[layer_idx]
        p.data.to(torch.device('cuda'))
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--num', default=10, type=int, help='number of models to ensemble')
parser.add_argument('--dataset', default='cifar', type=str, help='dataset')
parser.add_argument('--size', default='res18', type=str, help='param num')
parser.add_argument('--type', default='param', type=str, help='ensemble object')
parser.add_argument('--target_layer', default='linear')
parser.add_argument('--good', default='no')

args = parser.parse_args()
args.num_params_data = args.size
num_class = 10 if args.dataset == 'cifar' else 100
if args.size == 'res18':
    fc = classifier_res18(num_classes=num_class).cuda()
elif args.size == 'res50':
    fc = classifier_res50(num_classes=num_class).cuda()

featuredata = torch.load(os.path.join('../features', f"{args.dataset}_{args.size}/features_test.pt"))['features']
featuredata = torch.stack(featuredata)
featurelabel = torch.load(os.path.join('../features', f"{args.dataset}_{args.size}/features_test.pt"))['labels']
featurelabel = torch.tensor(featurelabel)
featureset = torch.utils.data.TensorDataset(featuredata,featurelabel)
featureloader = torch.utils.data.DataLoader(featureset,batch_size=255, shuffle=True, num_workers=0)

# well_fc = load_params(trained_params, fc)

best_params = []
load_good = args.good
if load_good == 'yes':
    accs = []
    path = '../parameters_onlyfc_test/cifar_res18/'
    files = os.listdir(path)
    for file in files:
        # import pdb;pdb.set_trace()
        param = torch.load(path+file)
        for one in param:
            one = torch.cat([p.data.reshape(-1) for p in one],0)
            fc = load_params(torch.squeeze(one), fc)
            acc,_= test_fc(fc.cuda(), featureloader)
            if acc > 92.6:
                accs.append(acc)
                best_params.append(one)  
    # best_params = torch.stack(best_params)
    print('=====================')
    print(f'total param number is {len(best_params)}')
    print(accs)
    print(np.max(np.array(accs)))

else:
    params = []
    trained_fcs = torch.load('../generated/cifar_res18_best20.pt')

    # well_fc = torch.load('../generated/cifar_res18_best1_new.pt')
# import pdb;pdb.set_trace()
# print(params.shape)
    for param in trained_fcs: #params is list
            for one in param:
                fc = load_params(torch.squeeze(one), fc)
                acc,_= test_fc(fc.cuda(), featureloader)
                best_params.append(one)
print('ensemble accuracy is:')
if args.type == 'param':
    fc = load_params(torch.squeeze(torch.mean(torch.stack(best_params),dim=0)), fc)
    acc= test_fc(fc.cuda(), featureloader)
elif args.type == 'logits':
    model_list = []
    for param in best_params:
        model_list.append(reverse_tomodel(torch.squeeze(param), fc))
    test_em(model_list, featureloader)
# ensem_acc = test_ensem_partial(args, best_params)