import torch
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import pdb
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

class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(3,5,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(6*6*5,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(-1, 6*6*5)
        x = self.dense(x)
        return x

net = Model()
# pdb.set_trace()
params = [p for p in net.parameters()]
for i in range(len(params)):
    print(i,params[i].shape)



