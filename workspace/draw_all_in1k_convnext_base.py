import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import timm

import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import dataset.in1k_convnext_base.for_draw_train as item
import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.datasets import CIFAR10 as Dataset
# from torch.utils.data import DataLoader
#
#
# test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
# test_dataset = Dataset(root="from_additional_config", train=False, download=True, transform=test_transform)
# loader = DataLoader(test_dataset, batch_size=200, shuffle=False,
#                                  num_workers=4, pin_memory=True)


loader = item.test_loader
model = item.model
test = item.test




checkpoint_path = "./dataset/in1k_convnext_base/checkpoint"
generated_path = "./dataset/in1k_convnext_base/generated"
tag = "in1k_convnext_base"
try:
    exec(sys.argv[1])
except:
    print("Please set noise_intensity=[xx.x, xx.x, xx.x]")
    noise_intensity = [0.03, 0.05, 0.10]




# load paths
checkpoint_items = [os.path.join(checkpoint_path, i) for i in os.listdir(checkpoint_path)]
generated_items = [os.path.join(generated_path, i) for i in os.listdir(generated_path)]
generated_items.sort()
import pdb; pdb.set_trace()
total_items = list(checkpoint_items) + list(generated_items)
num_checkpoint = len(checkpoint_items)
num_generated = len(generated_items)

criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def compute_wrong_indices(diction):
    model.load_state_dict(diction, strict=False)
    model.eval()
    # import pdb; pdb.set_trace()
    _, acc, all_targets, all_predicts = test(model=model, test_loader=loader, device=device)
    not_agreement = torch.logical_not(torch.eq(torch.tensor(all_targets), torch.tensor(all_predicts)))
    return not_agreement, acc

def compute_wrong_iou(a, b):
    inter = np.logical_and(a, b)
    union = np.logical_or(a, b)
    iou = np.sum(inter) / np.sum(union)
    return iou

# prepare evaluate
print("\n==> start evaluating..")
total_result_list = []
total_acc_list = []




# model = timm.create_model('resnet18', pretrained=True, num_classes=1000)
# model = model.to(device)
# compute checkpoint and generated
# load pretrained model
# pretrained_state = torch.load(
#     "/home/wangkai/cvpr_pdiff/p-diff-v2/dataset/in1k_resnet18/pretrained.pth")
# model.load_state_dict({key: value.to(torch.float32).to(device) for key, value in pretrained_state.items()},
#                       strict=False)
model = timm.create_model('convnext_base', pretrained=True, num_classes=1000)
model = model.to(device)
for i, item in enumerate(total_items):
    print(f"start: {i+1}/{len(total_items)}")
    item = torch.load(item, map_location='cpu')
    result, acc = compute_wrong_indices(item)
    result = result.numpy()
    total_result_list.append(result)
    total_acc_list.append(acc)

# compute noised
checkpoint_items_for_noise = checkpoint_items.copy()
random.shuffle(checkpoint_items_for_noise)
num_each_noised = len(checkpoint_items_for_noise) // len(noise_intensity)
num_noise_class = len(noise_intensity)
bias = 0

# load pretrained model
# pretrained_state = torch.load(
#     "/home/wangkai/cvpr_pdiff/p-diff-v2/dataset/in1k_resnet18/pretrained.pth")
# model.load_state_dict({key: value.to(torch.float32).to(device) for key, value in pretrained_state.items()},
#                       strict=False)
model = timm.create_model('convnext_base', pretrained=True, num_classes=1000)
model = model.to(device)
for this_noise_intensity in noise_intensity:
    for i in range(num_each_noised):
        i = i + bias
        print(f"testing noised: {i+1}/{num_each_noised * num_noise_class}")
        item = checkpoint_items_for_noise[i]
        item = torch.load(item, map_location="cpu")
        new_diction = {}
        for k, v in item.items():
            v += torch.randn_like(v) * this_noise_intensity
            new_diction[k] = v
        result, acc = compute_wrong_indices(new_diction)
        result = result.numpy()
        total_result_list.append(result)
        total_acc_list.append(acc)
    bias += num_each_noised




# compute iou_metrix
print("start computing IoU...")
total_num = num_checkpoint + num_generated + num_each_noised * num_noise_class
assert total_num == len(total_result_list), \
        f"total_num:{total_num}, len(total_result_list):{len(total_result_list)}"
iou_matrix = np.zeros(shape=[total_num, total_num])
for i in range(total_num):
    for j in range(total_num):
        iou = compute_wrong_iou(total_result_list[i], total_result_list[j])
        iou_matrix[i, j] = iou

# save result
df = pd.DataFrame(iou_matrix)
df.to_excel(f"./similarity_{tag}.xlsx", index=False)
print(f"finished Saving ./similarity_{tag}.xlsx!")




# print summary
print("\n\n===============================================\nSummary:")
print()
print("num_checkpoint:", num_checkpoint)
print("num_generated:", num_generated)
print(f"num_noised: {num_each_noised}x{num_noise_class}")
print()
origin_origin = iou_matrix[:num_checkpoint, :num_checkpoint]
origin_origin = (np.sum(origin_origin) - num_checkpoint) / (num_checkpoint * (num_checkpoint - 1))
print("origin-origin:", origin_origin)
generated_generated = iou_matrix[num_checkpoint:num_checkpoint + num_generated,
                                 num_checkpoint:num_checkpoint + num_generated]
generated_generated = (np.sum(generated_generated) - num_generated) / (num_generated * (num_generated - 1))
print("generated-generated:", generated_generated)
origin_generated = iou_matrix[num_checkpoint:num_checkpoint + num_generated, :num_checkpoint]
origin_generated = np.mean(origin_generated)
print("origin-generated:", origin_generated)
origin_generated_max = iou_matrix[num_checkpoint:num_checkpoint + num_generated, :num_checkpoint]
origin_generated_max = np.amax(origin_generated_max, axis=-1)
print("origin-generated(max):", origin_generated_max.mean())

# print noised
noised_max_list = []
this_start = num_checkpoint + num_generated
for this_noise_intensity in noise_intensity:
    print(f"\nnoise_intensity={this_noise_intensity}")
    noised_noised = iou_matrix[this_start:this_start + num_each_noised, this_start:this_start + num_each_noised]
    noised_noised = (np.sum(noised_noised) - num_each_noised) / (num_each_noised * (num_each_noised - 1))
    print("noised-noised:", noised_noised)
    origin_noised = iou_matrix[this_start:this_start + num_each_noised, :num_checkpoint]
    origin_noised = np.mean(origin_noised)
    print("origin-noised:", origin_noised)
    origin_noised_max = iou_matrix[this_start:this_start + num_each_noised, :num_checkpoint]
    origin_noised_max = np.amax(origin_noised_max, axis=-1)
    noised_max_list.append(origin_noised_max)
    print("origin-noised(max):", origin_noised_max.mean())
    this_start += num_each_noised



# final draw
print("\n==> start drawing..")
import seaborn as sns
import matplotlib.pyplot as plt
# origin
draw_origin_origin_max = np.amax(iou_matrix[:num_checkpoint, :num_checkpoint] - np.eye(num_checkpoint), axis=-1)
draw_origin_origin_acc = np.array(total_acc_list[:num_checkpoint])
sns.scatterplot(x=draw_origin_origin_max, y=draw_origin_origin_acc, label="origin")
# generated
draw_origin_generated_max = origin_generated_max
draw_origin_generated_acc = np.array(total_acc_list[num_checkpoint:num_checkpoint + num_generated])
sns.scatterplot(x=draw_origin_generated_max, y=draw_origin_generated_acc, label="generated")
# noised
this_start = num_checkpoint + num_generated
for i, this_noise_intensity in enumerate(noise_intensity):
    draw_origin_noised_max = noised_max_list[i]
    draw_origin_noised_acc = total_acc_list[this_start: this_start+num_each_noised]
    sns.scatterplot(x=draw_origin_noised_max, y=draw_origin_noised_acc, label=f"noise={this_noise_intensity:.4f}")
# draw
plt.savefig(f'plot_{tag}.png')
print(f"plot saved to plot_{tag}.png")
