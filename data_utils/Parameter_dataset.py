import os.path
from typing import Any, Callable, Optional, Tuple
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torch
import sys
sys.path.append('../')

class Parameters(VisionDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_model: int =1,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        print('Loading trained parameters dataset....')
        self.data_dir = root
        self.expert_files = []
        self.data = [] 
        self.num_model = num_model
        print(f'<<<<<<<number of expert models is {self.num_model}>>>>>>>')
        
        # pdb.set_trace()
        self.expert_files = os.listdir(self.data_dir)

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


class Parameters_partial(VisionDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_model: int =1,
        size='conv3',
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        print('Loading trained parameters dataset....')
        self.data_dir = root
        self.expert_files = []
        self.data = [] 
        self.num_model = num_model
        self.size = size
        print('model structure is',self.size)
        print(f'<<<<<<<number of expert models is {self.num_model}>>>>>>>')
        self.bns = []
        # import pdb;pdb.set_trace()
        self.expert_files = os.listdir(self.data_dir)
        # self.expert_files = ['parameters_8a3610c6-1.pt']
        for m in self.expert_files:
            models = self.data_dir + m
            buffers = torch.load(models) #一个buffers是n个teacher model的参数
            # if self.size != 'res18':
            for buffer in buffers: #一个buffer是一个 [p for p in model.parameter()]
                # param = buffer[-2:]  # 最后一个fc层的weight和bias
                "如果存储的已经只是fc了"
                param = buffer
                param = torch.cat([p.data.reshape(-1) for p in param], 0)
                self.data.append(param.cpu())
              
        print(f'<<<<<<<size of expert models is {self.data[0].shape[0]}>>>>>>>')
        batch = torch.stack(self.data)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)
        self.mean = mean
        self.std = std
        self.data = batch[0:self.num_model]
        # import pdb;pdb.set_trace()

    # def de_norm_param(self, param):
    #     return param * self.std + self.mean


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img = self.data[index%self.num_model]
            # import pdb;pdb.set_trace()
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
    
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.MNIST(root="../../../datasets/",
                           transform = transform,
                           train = False)
    root = '/home/pangshengyuan/code/pytorch-cifar-master/res/mnist'
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)
    set = Parameters(root)      
    loader = DataLoader(set, batch_size=4, shuffle=True)
    # net = Model()
    # num_params = sum([np.prod(p.size()) for p in (net.parameters())])
    # print(f'parameters number is {num_params}')
    model = MNIST_Model().cuda()
    for i, x in enumerate(loader):
        num_params = sum([np.prod(p.size()) for p in (x[0])])
        print(num_params) #5066
        print(x.shape)    # [batchsize, 5066]
        model = reverse_tomodel(x[0], model).cuda()
        acc, loss = test(model, testloader)
        print(f'test accuracy is {acc}')



