import torch
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import pdb
from main_mnist import Model, test
transform = transforms.Compose([transforms.ToTensor(),])
                            #    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])


testset = torchvision.datasets.MNIST(root="../../../datasets/",
                           download= False,
                           transform = transform,
                           train = False)
testloader = torch.utils.data.DataLoader(
testset, batch_size=100, shuffle=False, num_workers=0)
net = Model().cpu()
layer_idx = 0
path = '../parameters/mnist/parameters_0.pt'
params = torch.load(path)[1]
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
def swap(lst, index1, index2):
    # Check if the indices are valid
    if index1 < 0 or index1 >= len(lst) or index2 < 0 or index2 >= len(lst):
        print("Invalid indices.")
        return
    
    # Swap the elements
    temp = lst[index1]
    lst[index1] = lst[index2]
    lst[index2] = temp


# pdb.set_trace()
for p in net.parameters(): 
    p.data = params[layer_idx]
    p.data#.to(torch.device('cuda'))
    p.data.requires_grad_(True)
    layer_idx += 1
test(net, testloader)

# layer 0
# tmp = params[0][0].clone()
# params[0][0] = params[0][1]
# params[0][1] = tmp

# # layer 1
# tmp = params[1][0].clone()
# params[1][0] = params[1][1]
# params[1][1] = tmp

# # layer 2
# for i in range(4):
#     tmp = params[2][i][0].clone()
#     params[2][i][0] = params[2][i][1]
#     params[2][i][1] = tmp

# layer_idx = 0
# for p in net.parameters(): 
#     p.data = params[layer_idx]
#     p.data.to(torch.device('cuda'))
#     p.data.requires_grad_(True)
#     layer_idx += 1


#==========  如果不对layer2做reorder 则layer3不变且前后相互独立
# layer 4
# tmp = params[4][0].clone()
# params[4][0] = params[4][1]
# params[4][1] = tmp

# # layer 5
# tmp = params[5][0].clone()
# params[5][0] = params[5][1]
# params[5][1] = tmp

# # layer 6
# tmp = params[6][:,0].clone()
# params[6][:,0] = params[6][:,1]
# params[6][:,1] = tmp

# layer 7 bias in the last layer (dont change due to the fixed classes)
# tmp = params[7][0].clone()
# params[7][0] = params[7][1]
# params[7][1] = tmp

# test(net.cuda(), testloader)
