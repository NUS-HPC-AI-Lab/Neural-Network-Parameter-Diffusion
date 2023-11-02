import torch
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import pdb
import numpy as np
from utils.main_mnist import Model, test
# from main_mnist import Model, test

"""
params: len=8的list
0: torch[2,1,3,3]  
1: torch[2]
2: torch[4,2,3,3]
3: torch[4]
4: torch[32,144]
5: torch[32]
6: torch[10,32]
7: torch[10]
"""
def reorder(layer: torch.Tensor):
    """
    input: layer is a layer of parameters
    return: reordered layer

    """
    scores = [param.sum().item() for param in layer]
    index = [i for i in range(len(scores))]
    n = len(scores)
    flag = False
    for i in range(n):
        for j in range(n-i-1):
            if scores[j] < scores[j+1]:
                # pdb.set_trace()
                # print('check')
                layer[j], layer[j+1] = swap(layer[j], layer[j+1])
                flag = True
                scores[j], scores[j+1] = scores[j+1], scores[j]
                index[j], index[j+1] = index[j+1], index[j]
    return layer, flag, index

def swap(tensor1,tensor2):
    tmp = tensor1.clone()
    tensor1 = tensor2
    tensor2 = tmp
    return tensor1, tensor2
 

def reorder_model(params):
    """
    inputs: params 一个list 每一个element是一层的tensor
    """
    ## reorder layer 0
    params[0], flag, index = reorder(params[0])
    # print(flag)

    #layer1:bias  layer2:第二层卷积核内通道随着变
    params[1] = params[1][index]
    for i in range(4):
            params[2][i] = params[2][i][index]

    ## reorder layer2：第二层卷积核之间重排序
    params[2], flag, index = reorder(params[2])

    ## layer3: 第二层的bias  layer4:第一层全连接层 随着变 
    params[3] = params[3][index]
    # print(index)
    new = []
    for i in index:
        tmp = [j for j in range(i*36,(i+1)*36)]
        new += tmp
    # index = [j for j in range(i*36,(i+1)*36) for i in index]
    # print(new)
    params[4][:,] = params[4][:,new]

    ## reorder layer4:
    params[4], flag , index = reorder(params[4])
    ## layer5: 第一层全连接层的输出层的bias  layer6:下一层的输入 随着变
    params[5] = params[5][index]
    params[6][:,] = params[6][:,index]
    # test(net, testloader)
    return params



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),])
                            #    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])


    testset = torchvision.datasets.MNIST(root="./data/",
                            download= False,
                            transform = transform,
                            train = False,)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)
    layer_idx = 0
    path = '../parameters/mnist/parameters_0.pt'

    for j in range(10):
        params = torch.load(path)[j]
        params = reorder_model(params)
        net = Model()
        layer_idx = 0
        
        for p in net.parameters(): 
            # pdb.set_trace()
            p.data = params[layer_idx]
            # p.data.to(torch.device('cuda'))
            p.data.requires_grad_(True)
            layer_idx += 1
        test(net, testloader)

     
    