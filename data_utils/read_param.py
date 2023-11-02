import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pdb
# class Model(torch.nn.Module):
    
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2,kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Dropout(p=0.5),
#                                          torch.nn.Linear(1024, 10))
#     def forward(self, x):
#         x = self.conv1(x)
#         #x = self.conv2(x)
#         x = x.view(-1, 14*14*128)
#         x = self.dense(x)
#         return x


class ParameterDataset(Dataset):
    
        def __init__(self, dir) -> None:
            super(ParameterDataset, self).__init__()
            print('Loading trained parameters dataset....')
            self.data_dir = dir #'/share/psy/mnist'
            self.expert_files = []

        def __len__(self):
            n = 0
            while os.path.exists(os.path.join(self.data_dir, "parameters_{}.pt".format(n))):
                self.expert_files.append(os.path.join(self.data_dir, "parameters_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(self.data_dir))
            return 100*n

        def __getitem__(self, index):
            buffer_index = index//100
            model_index = index%100
            buffer = torch.load(self.expert_files[buffer_index]) #一个buffer是100个teacher model的参数 
            model = buffer[model_index]
            param = torch.cat([p.data.reshape(-1) for p in model], 0)
            print('check', param.shape)
            return torch.cat([p.data.reshape(-1) for p in model], 0)

set = ParameterDataset('/home/pangshengyuan/code/pytorch-cifar-master/res/mnist')      
loader = DataLoader(set, batch_size=4, shuffle=True)
# net = Model()
# num_params = sum([np.prod(p.size()) for p in (net.parameters())])
# print(f'parameters number is {num_params}')
for i, x in enumerate(loader):
    num_params = sum([np.prod(p.size()) for p in (x[0])])
    # print(num_params)
  
    # pdb.set_trace()


# expert_files = []
# n = 0

# while os.path.exists(os.path.join('/share/psy/mnist', "parameters_{}.pt".format(n))):
#     expert_files.append(os.path.join('/share/psy/mnist', "parameters_{}.pt".format(n)))
#     n += 1
# if n == 0:
#     raise AssertionError("No buffers detected at {}".format('/share/psy/mnist'))
# print(f'expert_files is {expert_files}')
# file_idx = 0
# expert_idx = 0

# final_parameters = []
# # for i in range(len(expert_files)):
# for i in range(len(expert_files)):

#     print("loading file {}".format(expert_files[i])) 
#     buffer = torch.load(expert_files[i]) #一个buffer是100个teacher model的参数 
#     for j in range(len(buffer)):
#         final_parameters.append(torch.cat([p.data.reshape(-1) for p in buffer[j]], 0))



# dist = []
# net = Model()
# num_params = sum([np.prod(p.size()) for p in (net.parameters())])
# num_params = sum([np.prod(p.size()) for p in (final_parameters[0])])

# for i in range(len(final_parameters)):
#  for j in range(i+1, len(final_parameters)):

#     param_distance = torch.tensor(0.0)
#     param_distance += torch.nn.functional.mse_loss(final_parameters[i], final_parameters[j], reduction="sum") #目标终点距离
#     param_distance /= num_params
#     dist.append(param_distance.detach().cpu().numpy())
# dist = np.array(dist)
# print(f'The mean distance is {np.mean(dist)}')
