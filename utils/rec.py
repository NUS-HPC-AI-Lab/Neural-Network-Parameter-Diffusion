import torch
# from datautils.mycifar import CIFAR10 as mycifarCIFAR10

# import data_utils.mycifar as mycifar
import os.path
from typing import Any, Callable, Optional, Tuple
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F


import os
class MNIST_5066(torch.nn.Module):
    
    def __init__(self):
        super(MNIST_5066, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(6*6*4,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(-1, 6*6*4)
        x = self.dense(x)
        return x
class CNNParameters_Mnist(VisionDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_model: int =0,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        print('Loading trained parameters dataset....')
        self.data_dir = root
        self.expert_files = []
        self.data = [] 

        
        # pdb.set_trace()
        self.expert_files = os.listdir(self.data_dir)
        if num_model == 0:
            self.num_model = len(self.expert_files)*10
        else:
            self.num_model = num_model
        print(f'<<<<<<<number of expert models is {self.num_model}>>>>>>>')
        for m in self.expert_files:
            models = self.data_dir + m
            buffers = torch.load(models) #一个buffers是n个teacher model的参数 
            for buffer in buffers: #一个buffer是一个 [p for p in model.parameter()]
                # param = reorder_model(buffer)
                param = buffer
                param = torch.cat([p.data.reshape(-1) for p in param], 0)
                self.data.append(param)
        print(f'<<<<<<<size of expert models is {self.data[0].shape[0]}>>>>>>>')
        batch = torch.stack(self.data)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)
        self.mean = mean
        self.std = std
        self.data = batch[0:self.num_model]
        # import pdb; pdb.set_trace()
        print(self.data.shape)

    # def de_norm_param(self, param):
    #     return param * self.std + self.mean


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img = self.data[index%self.num_model]
        return img

    def __len__(self) -> int:
        return 5000

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    

def test_acc(model, test_loader):
    model.eval()
    model = model.cuda()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader_mnist:
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    print('\n Test_set on Mnist: Average loss: {:.4f}, Accuracy: {:.4f}\n'
        .format(test_loss, acc))
    return acc, test_loss

teacher = MNIST_5066()
class Latent_AE_cnn_test(nn.Module):
    def __init__(
        self,
        in_dim,
        time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 4 #就是stride size
        self.kernal_size = 4
        self.channel_list = [4, 8, 16, 32]
        self.real_input_dim = (
            int(in_dim / self.fold_rate**4 + 1) * self.fold_rate**4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**4),
            nn.ConvTranspose1d(
                self.channel_list[3], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**4),
            nn.Conv1d(self.channel_list[3], self.channel_list[2], self.kernal_size, stride=1, padding=self.fold_rate-1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**3),
            nn.ConvTranspose1d(
                self.channel_list[2], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**3),
            nn.Conv1d(self.channel_list[2], self.channel_list[1], self.kernal_size, stride=1, padding=self.fold_rate-1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**2),
            nn.ConvTranspose1d(
                self.channel_list[1], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate**2),
            nn.Conv1d(self.channel_list[1], self.channel_list[0], self.kernal_size, stride=1, padding=self.fold_rate-1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_dec1 = self.dec1(emb_enc4) 
        emb_dec2 = self.dec2(emb_dec1) 
        emb_dec3 = self.dec3(emb_dec2) 
        emb_dec4 = self.dec4(emb_dec3)[:,:,:input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)
        
        return emb_enc4
    
    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:,:,:self.in_dim]

        return emb_dec4
testloader_mnist =     torch.utils.data.DataLoader(torchvision.datasets.MNIST(root="./data/",
                            transform = transforms.Compose([transforms.ToTensor(),]),
                            train = False), batch_size=100, shuffle=False, num_workers=0)
def reverse_tomodel( flattened, model):
        flattened = torch.squeeze(flattened)
        example_parameters = [p for p in model.parameters()]
        len = 0
        reversed_params = []
        # layer = 0
        # import pdb; pdb.set_trace()
        for p in example_parameters:
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
def train(ae, trainloader):
    for batch_data in trainloader:
            # 正向传播
            outputs = ae(batch_data)
            loss = criterion(outputs, batch_data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss
            
def test(ae, testloader):
    count =0 
    for data in testloader:
        recon = ae(data)
        # import pdb;pdb.set_trace()
        recon = reverse_tomodel(recon,teacher)
        count += 1
        if count ==5:
            break
        acc , loss = test_acc(recon, test_loader)
if __name__ == '__main__':
    ae = Latent_AE_cnn_test(in_dim=5066)
    dataset = CNNParameters_Mnist(root='../parameters/mnist_5066/')
    
    # 创建数据加载器
    batch_size = 32
    trainset, testset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)

    
    # 定义优化器和损失函数
    learning_rate = 0.000001
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 设置训练参数
    num_epochs = 10000
    
    # 开始训练
    for epoch in range(num_epochs):
        print(f'<<<<<<<<<<<<<<<<<<<<<Epoch {epoch}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        loss = train(ae, train_loader)
        if epoch % 50 ==0:
            test(ae, test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # test

    

