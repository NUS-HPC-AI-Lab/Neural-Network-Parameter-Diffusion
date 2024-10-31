import torch
import einops
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
import math
import random
import json
from abc import ABC


def pad_to_length(x, common_factor, **config):
    if x.numel() % common_factor == 0:
        return x.flatten()
    # print(f"padding {x.shape} according to {common_factor}")
    full_length = (x.numel() // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    padding = torch.full([padding_length, ], dtype=x.dtype, device=x.device, fill_value=config["fill_value"])
    x = torch.cat((x.flatten(), padding), dim=0)
    return x

def layer_to_token(x, common_factor, **config):
    if config["granularity"] == 2:  # split by output
        if x.numel() <= common_factor:
            return pad_to_length(x.flatten(), common_factor, **config)[None]
        dim2 = x[0].numel()
        dim1 = x.shape[0]
        if dim2 <= common_factor:
            i = int(dim1 / (common_factor / dim2))
            while True:
                if dim1 % i == 0 and dim2 * (dim1 // i) <= common_factor:
                    output = x.view(-1, dim2 * (dim1 // i))
                    output = [pad_to_length(item, common_factor, **config) for item in output]
                    return torch.stack(output, dim=0)
                i += 1
        else:  # dim2 > common_factor
            output = [layer_to_token(item, common_factor, **config) for item in x]
            return torch.cat(output, dim=0)
    elif config["granularity"] == 1:  # split by layer
        return pad_to_length(x.flatten(), common_factor, **config).view(-1, common_factor)
    elif config["granularity"] == 0:  # flatten directly
        return x.flatten()
    else:  # NotImplementedError
        raise NotImplementedError("granularity: 0: flatten directly, 1: split by layer, 2: split by output dim")


def token_to_layer(tokens, shape, **config):
    common_factor = tokens.shape[-1]
    if config["granularity"] == 2:  # split by output
        num_element = math.prod(shape)
        if num_element <= common_factor:
            param = tokens[0][:num_element].view(shape)
            tokens = tokens[1:]
            return param, tokens
        dim2 = num_element // shape[0]
        dim1 = shape[0]
        if dim2 <= common_factor:
            i = int(dim1 / (common_factor / dim2))
            while True:
                if dim1 % i == 0 and dim2 * (dim1 // i) <= common_factor:
                    item_per_token = dim2 * (dim1 // i)
                    length = num_element // item_per_token
                    output = [item[:item_per_token] for item in tokens[:length]]
                    param = torch.cat(output, dim=0).view(shape)
                    tokens = tokens[length:]
                    return param, tokens
                i += 1
        else:  # dim2 > common_factor
            output = []
            for i in range(shape[0]):
                param, tokens = token_to_layer(tokens, shape[1:], **config)
                output.append(param.flatten())
            param = torch.cat(output, dim=0).view(shape)
            return param, tokens
    elif config["granularity"] == 1:  # split by layer
        num_element = math.prod(shape)
        token_num = num_element // common_factor if num_element % common_factor == 0 \
                else num_element // common_factor + 1
        param = tokens.flatten()[:num_element].view(shape)
        tokens = tokens[token_num:]
        return param, tokens
    elif config["granularity"] == 0:  # flatten directly
        num_element = math.prod(shape)
        param = tokens.flatten()[:num_element].view(shape)
        tokens = pad_to_length(tokens.flatten()[num_element:],
                common_factor, fill_value=torch.nan).view(-1, common_factor)
        return param, tokens
    else:  # NotImplementedError
        raise NotImplementedError("granularity: 0: flatten directly, 1: split by layer, 2: split by output dim")


def positional_embedding_2d(dim1, dim2, d_model):
    assert d_model % 4 == 0, f"Cannot use sin/cos positional encoding with odd dimension {d_model}"
    pe = torch.zeros(d_model, dim1, dim2)
    d_model = int(d_model / 2)  # Each dimension use half of d_model
    div_term = torch.exp(torch.arange(0., d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., dim2).unsqueeze(1)
    pos_h = torch.arange(0., dim1).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, dim1, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, dim1, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, dim2)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, dim2)
    return pe.permute(1, 2, 0)


def positional_embedding_1d(dim1, d_model):
    pe = torch.zeros(dim1, d_model)
    position = torch.arange(0, dim1, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe




class BaseDataset(Dataset, ABC):
    data_path = None
    generated_path = None
    test_command = None
    config = {
        "fill_value": torch.nan,
        "granularity": 1,  # 0: flatten directly, 1: split by layer, 2: split by output
        "pe_granularity": 2,  # 0: no embedding, 1: 1d embedding, 2: 2d embedding
    }

    def __init__(self, checkpoint_path=None, dim_per_token=8192, **kwargs):
        self.config.update(kwargs)
        checkpoint_path = self.data_path if checkpoint_path is None else checkpoint_path
        # import pdb; pdb.set_trace()
        assert os.path.exists(checkpoint_path)
        self.dim_per_token = dim_per_token
        self.structure = None  # set in get_structure()
        self.sequence_length = None  # set in get_structure()
        # load checkpoint_list
        checkpoint_list = os.listdir(checkpoint_path)
        self.checkpoint_list = list([os.path.join(checkpoint_path, item) for item in checkpoint_list])
        self.length = self.real_length = len(self.checkpoint_list)
        self.set_infinite_dataset()
        self.get_structure()
        # other kwargs

    def get_structure(self):
        # get structure
        checkpoint_list = self.checkpoint_list
        structures = [{} for _ in range(len(checkpoint_list))]
        for i, checkpoint in enumerate(checkpoint_list):
            diction = torch.load(checkpoint, map_location="cpu")
            for key, value in diction.items():
                if ("num_batches_tracked" in key) or (value.numel() == 1) or not torch.is_floating_point(value):
                    structures[i][key] = (value.shape, value, None)
                elif "running_var" in key:
                    pre_mean = value.mean() * 0.95
                    value = torch.log(value / pre_mean + 0.05)
                    structures[i][key] = (value.shape, pre_mean, value.mean(), value.std())
                else:  # conv & linear
                    structures[i][key] = (value.shape, value.mean(), value.std())
        final_structure = {}
        structure_diction = torch.load(checkpoint_list[0], map_location="cpu")
        for key, param in structure_diction.items():
            if ("num_batches_tracked" in key) or (param.numel() == 1) or not torch.is_floating_point(param):
                final_structure[key] = (param.shape, param, None)
            elif "running_var" in key:
                value = [param.shape, 0., 0., 0.]
                for structure in structures:
                    for i in [1, 2, 3]:
                        value[i] += structure[key][i]
                for i in [1, 2, 3]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
            else:  # conv & linear
                value = [param.shape, 0., 0.]
                for structure in structures:
                    for i in [1, 2]:
                        # import pdb; pdb.set_trace()
                        value[i] += structure[key][i]
                for i in [1, 2]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
        self.structure = final_structure
        # get sequence_length
        param = self.preprocess(structure_diction)
        self.sequence_length = param.size(0)

    def set_infinite_dataset(self, max_num=None):
        if max_num is None:
            max_num = self.length * 1000000
        self.length = max_num
        return self

    @property
    def max_permutation_state(self):
        return self.real_length

    def get_position_embedding(self, positional_embedding_dim=None):
        if positional_embedding_dim is None:
            positional_embedding_dim = self.dim_per_token // 2
        assert self.structure is not None, "run get_structure before get_position_embedding"
        if self.config["pe_granularity"] == 2:
            print("Use 2d positional embedding")
            positional_embedding_index = []
            for key, item in self.structure.items():
                if ("num_batches_tracked" in key) or (item[-1] is None):
                    continue
                else:  # conv & linear
                    shape, *_ = item
                fake_param = torch.ones(size=shape)
                fake_param = layer_to_token(fake_param, self.dim_per_token, **self.config)
                positional_embedding_index.append(list(range(fake_param.size(0))))
            dim1 = len(positional_embedding_index)
            dim2 = max([len(token_per_layer) for token_per_layer in positional_embedding_index])
            full_pe = positional_embedding_2d(dim1, dim2, positional_embedding_dim)
            positional_embedding = []
            for layer_index, token_indexes in enumerate(positional_embedding_index):
                for token_index in token_indexes:
                    this_pe = full_pe[layer_index, token_index]
                    positional_embedding.append(this_pe)
            positional_embedding = torch.stack(positional_embedding)
            return positional_embedding
        elif self.config["pe_granularity"] == 1:
            print("Use 1d positional embedding")
            return positional_embedding_1d(self.sequence_length, positional_embedding_dim)
        elif self.config["pe_granularity"] == 0:
            print("Not use positional embedding")
            return torch.zeros_like(self.__getitem__(0))
        else:  # NotImplementedError
            raise NotImplementedError("pe_granularity: 0: no embedding, 1: 1d embedding, 2: 2d embedding")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        param = self.preprocess(diction)
        return param, index

    def save_params(self, params, save_path):
        diction = self.postprocess(params.cpu().to(torch.float32))
        torch.save(diction, save_path)

    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        param_list = []
        for key, value in diction.items():
            if ("num_batches_tracked" in key) or (value.numel() == 1) or not torch.is_floating_point(value):
                continue
            elif "running_var" in key:
                shape, pre_mean, mean, std = self.structure[key]
                value = torch.log(value / pre_mean + 0.05)
            else:  # normal
                shape, mean, std = self.structure[key]
            value = (value - mean) / std
            value = layer_to_token(value, self.dim_per_token, **self.config)
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
        if self.config["granularity"] == 0:  # padding directly process tail
            param = pad_to_length(param, self.dim_per_token, **self.config).view(-1, self.dim_per_token)
        # print("Sequence length:", param.size(0))
        return param.to(torch.float32)

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        diction = {}
        params = params if len(params.shape) == 2 else params.squeeze(0)
        for key, item in self.structure.items():
            if ("num_batches_tracked" in key) or (item[-1] is None):
                shape, mean, std = item
                diction[key] = mean
                continue
            elif "running_var" in key:
                shape, pre_mean, mean, std = item
            else:  # conv & linear
                shape, mean, std = item
            this_param, params = token_to_layer(params, shape, **self.config)
            this_param = this_param * std + mean
            if "running_var" in key:
                this_param = torch.clip(torch.exp(this_param) - 0.05, min=0.001) * pre_mean
            diction[key] = this_param
        return diction




class Cifar10_CNNSmall(BaseDataset):
    data_path = "./dataset/cifar10_cnnsmall/checkpoint"
    generated_path = "./dataset/cifar10_cnnsmall/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_cnnsmall/test.py " + \
                   "./dataset/cifar10_cnnsmall/generated/generated_model.pth"

class Cifar100_ResNet18BN(BaseDataset):
    data_path = "./dataset/cifar100_resnet18/checkpoint"
    generated_path = "./dataset/cifar100_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_resnet18/test.py " + \
                   "./dataset/cifar100_resnet18/generated/generated_model.pth"

class Cifar10_ResNet18BN(BaseDataset):
    data_path = "./dataset/cifar10_resnet18/checkpoint"
    generated_path = "./dataset/cifar10_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet18/test.py " + \
                   "./dataset/cifar10_resnet18/generated/generated_model.pth"

class Cifar10_ResNet50BN(BaseDataset):
    data_path = "./dataset/cifar10_resnet50/checkpoint"
    generated_path = "./dataset/cifar10_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet50/test.py " + \
                   "./dataset/cifar10_resnet50/generated/generated_model.pth"


class Cifar10_Vit_Tiny(BaseDataset):
    data_path = "./dataset/cifar10_vit_tiny/checkpoint"
    generated_path = "./dataset/cifar10_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_vit_tiny/test.py " + \
                   "./dataset/cifar10_vit_tiny/generated/generated_model.pth"

class Cifar10_Vit_Base(BaseDataset):
    data_path = "./dataset/cifar10_vit_base/checkpoint"
    generated_path = "./dataset/cifar10_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_vit_base/test.py " + \
                   "./dataset/cifar10_vit_base/generated/generated_model.pth"

class Cifar10_ConvNeXt_Tiny(BaseDataset):
    data_path = "./dataset/cifar10_ConvNeXt_T/checkpoint"
    generated_path = "./dataset/cifar10_ConvNeXt_T/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_ConvNeXt_T/test.py " + \
                   "./dataset/cifar10_ConvNeXt_T/generated/generated_model.pth"

class Cifar10_ConvNeXt_Base(BaseDataset):
    data_path = "./dataset/cifar10_ConvNeXt_B/checkpoint"
    generated_path = "./dataset/cifar10_ConvNeXt_B/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_ConvNeXt_B/test.py " + \
                   "./dataset/cifar10_ConvNeXt_B/generated/generated_model.pth"

class Cifar10_ConvNet3(BaseDataset):
    data_path = "./dataset/cifar10_convnet3/checkpoint"
    generated_path = "./dataset/cifar10_convnet3/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_convnet3/test.py " + \
                   "./dataset/cifar10_convnet3/generated/generated_model.pth"



# class Cifar100_ConvNeXt_Tiny(BaseDataset):
#     data_path = "./dataset/cifar100_ConvNeXt_T/checkpoint"
#     generated_path = "./dataset/cifar100_ConvNeXt_T/generated/generated_model.pth"
#     test_command = "python ./dataset/cifar100_ConvNeXt_T/test.py " + \
#                    "./dataset/cifar100_ConvNeXt_T/generated/generated_model.pth"

class Cifar100_Vit_Tiny(BaseDataset):
    data_path = "./dataset/cifar100_vit_tiny/checkpoint"
    generated_path = "./dataset/cifar100_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_vit_tiny/test.py " + \
                   "./dataset/cifar100_vit_tiny/generated/generated_model.pth"

class Cifar100_Vit_Base(BaseDataset):
    data_path = "./dataset/cifar100_vit_base/checkpoint"
    generated_path = "./dataset/cifar100_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_vit_base/test.py " + \
                   "./dataset/cifar100_vit_base/generated/generated_model.pth"

class Cifar100_ConvNeXt_Tiny(BaseDataset):
    data_path = "./dataset/cifar100_convnext_tiny/checkpoint"
    generated_path = "./dataset/cifar100_convnext_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_convnext_tiny/test.py " + \
                   "./dataset/cifar100_convnext_tiny/generated/generated_model.pth"

class Cifar100_ResNet50(BaseDataset):
    data_path = "./dataset/cifar100_resnet50/checkpoint"
    generated_path = "./dataset/cifar100_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_resnet50/test.py " + \
                   "./dataset/cifar100_resnet50/generated/generated_model.pth"

class Cifar100_ResNet18(BaseDataset):
    data_path = "./dataset/cifar100_resnet18/checkpoint"
    generated_path = "./dataset/cifar100_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_resnet18/test.py " + \
                   "./dataset/cifar100_resnet18/generated/generated_model.pth"

class Cifar100_ConvNeXt_Base(BaseDataset):
    data_path = "./dataset/cifar100_convnext_base/checkpoint"
    generated_path = "./dataset/cifar100_convnext_base/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_convnext_base/test.py " + \
                   "./dataset/cifar100_convnext_base/generated/generated_model.pth"


class in1k_ResNet18(BaseDataset):
    data_path = "./dataset/in1k_resnet18/checkpoint"
    generated_path = "./dataset/in1k_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_resnet18/test.py " + \
                   "./dataset/in1k_resnet18/generated/generated_model.pth"

class in1k_ResNet50(BaseDataset):
    data_path = "./dataset/in1k_resnet50/checkpoint"
    generated_path = "./dataset/in1k_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_resnet50/test.py " + \
                   "./dataset/in1k_resnet50/generated/generated_model.pth"

class in1k_ViT_Tiny(BaseDataset):
    data_path = "./dataset/in1k_vit_tiny/checkpoint"
    generated_path = "./dataset/in1k_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_vit_tiny/test.py " + \
                   "./dataset/in1k_vit_tiny/generated/generated_model.pth"

class in1k_ViT_Base(BaseDataset):
    data_path = "./dataset/in1k_vit_base/checkpoint"
    generated_path = "./dataset/in1k_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_vit_base/test.py " + \
                   "./dataset/in1k_vit_base/generated/generated_model.pth"

class in1k_convnext_tiny(BaseDataset):
    data_path = "./dataset/in1k_convnext_tiny/checkpoint"
    generated_path = "./dataset/in1k_convnext_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_convnext_tiny/test.py " + \
                   "./dataset/in1k_convnext_tiny/generated/generated_model.pth"

class in1k_convnext_base(BaseDataset):
    data_path = "./dataset/in1k_convnext_base/checkpoint"
    generated_path = "./dataset/in1k_convnext_base/generated/generated_model.pth"
    test_command = "python ./dataset/in1k_convnext_base/test.py " + \
                   "./dataset/in1k_convnext_base/generated/generated_model.pth"

class STL10_ResNet18(BaseDataset):
    data_path = "./dataset/stl10_resnet18/checkpoint"
    generated_path = "./dataset/stl10_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_resnet18/test.py " + \
                   "./dataset/stl10_resnet18/generated/generated_model.pth"

class STL10_ResNet50(BaseDataset):
    data_path = "./dataset/stl10_resnet50/checkpoint"
    generated_path = "./dataset/stl10_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_resnet50/test.py " + \
                   "./dataset/stl10_resnet50/generated/generated_model.pth"

class STL10_ViT_Tiny(BaseDataset):
    data_path = "./dataset/stl10_vit_tiny/checkpoint"
    generated_path = "./dataset/stl10_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_vit_tiny/test.py " + \
                   "./dataset/stl10_vit_tiny/generated/generated_model.pth"

class STL10_ViT_Base(BaseDataset):
    data_path = "./dataset/stl10_vit_base/checkpoint"
    generated_path = "./dataset/stl10_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_vit_base/test.py " + \
                   "./dataset/stl10_vit_base/generated/generated_model.pth"

class STL10_ConvNext_Tiny(BaseDataset):
    data_path = "./dataset/stl10_convnext_tiny/checkpoint"
    generated_path = "./dataset/stl10_convnext_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_convnext_tiny/test.py " + \
                   "./dataset/stl10_convnext_tiny/generated/generated_model.pth"

class STL10_ConvNext_Base(BaseDataset):
    data_path = "./dataset/stl10_convnext_base/checkpoint"
    generated_path = "./dataset/stl10_convnext_base/generated/generated_model.pth"
    test_command = "python ./dataset/stl10_convnext_base/test.py " + \
                   "./dataset/stl10_convnext_base/generated/generated_model.pth"


class pets_ResNet18(BaseDataset):
    data_path = "./dataset/pets_resnet18/checkpoint"
    generated_path = "./dataset/pets_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/pets_resnet18/test.py " + \
                   "./dataset/pets_resnet18/generated/generated_model.pth"

class pets_ResNet50(BaseDataset):
    data_path = "./dataset/pets_resnet50/checkpoint"
    generated_path = "./dataset/pets_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/pets_resnet50/test.py " + \
                   "./dataset/pets_resnet50/generated/generated_model.pth"

class pets_ViT_Tiny(BaseDataset):
    data_path = "./dataset/pets_vit_tiny/checkpoint"
    generated_path = "./dataset/pets_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/pets_vit_tiny/test.py " + \
                   "./dataset/pets_vit_tiny/generated/generated_model.pth"

class pets_ViT_Base(BaseDataset):
    data_path = "./dataset/pets_vit_base/checkpoint"
    generated_path = "./dataset/pets_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/pets_vit_base/test.py " + \
                   "./dataset/pets_vit_base/generated/generated_model.pth"

class pets_ConvNext_Tiny(BaseDataset):
    data_path = "./dataset/pets_convnext_tiny/checkpoint"
    generated_path = "./dataset/pets_convnext_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/pets_convnext_tiny/test.py " + \
                   "./dataset/pets_convnext_tiny/generated/generated_model.pth"     

class pets_ConvNext_Base(BaseDataset):
    data_path = "./dataset/pets_convnext_base/checkpoint"
    generated_path = "./dataset/pets_convnext_base/generated/generated_model.pth"
    test_command = "python ./dataset/pets_convnext_base/test.py " + \
                   "./dataset/pets_convnext_base/generated/generated_model.pth"

class flowers_ResNet18(BaseDataset):
    data_path = "./dataset/flowers_resnet18/checkpoint"
    generated_path = "./dataset/flowers_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_resnet18/test.py " + \
                   "./dataset/flowers_resnet18/generated/generated_model.pth"

class flowers_ResNet50(BaseDataset):
    data_path = "./dataset/flowers_resnet50/checkpoint"
    generated_path = "./dataset/flowers_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_resnet50/test.py " + \
                   "./dataset/flowers_resnet50/generated/generated_model.pth"

class flowers_ViT_Tiny(BaseDataset):
    data_path = "./dataset/flowers_vit_tiny/checkpoint"
    generated_path = "./dataset/flowers_vit_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_vit_tiny/test.py " + \
                   "./dataset/flowers_vit_tiny/generated/generated_model.pth"
                   
class flowers_ViT_Base(BaseDataset):
    data_path = "./dataset/flowers_vit_base/checkpoint"
    generated_path = "./dataset/flowers_vit_base/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_vit_base/test.py " + \
                   "./dataset/flowers_vit_base/generated/generated_model.pth"


class flowers_ConvNext_Tiny(BaseDataset):
    data_path = "./dataset/flowers_convnext_tiny/checkpoint"
    generated_path = "./dataset/flowers_convnext_tiny/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_convnext_tiny/test.py " + \
                   "./dataset/flowers_convnext_tiny/generated/generated_model.pth"


class flowers_ConvNext_Base(BaseDataset):
    data_path = "./dataset/flowers_convnext_base/checkpoint"
    generated_path = "./dataset/flowers_convnext_base/generated/generated_model.pth"
    test_command = "python ./dataset/flowers_convnext_base/test.py " + \
                   "./dataset/flowers_convnext_base/generated/generated_model.pth"
                   
                   
class Cifar100_ResNet18_ablation_ft_lr_001(BaseDataset):
    data_path = "./dataset/ablation_cifar100_resnet18_different_ft_lr/checkpoint"
    generated_path = "./dataset/ablation_cifar100_resnet18_different_ft_lr/generated/generated_model.pth"
    test_command = "python ./dataset/ablation_cifar100_resnet18_different_ft_lr/test.py " + \
                   "./dataset/ablation_cifar100_resnet18_different_ft_lr/generated/generated_model.pth"