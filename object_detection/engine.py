import math
import pdb
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from torchvision.ops import nms

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from pycocotools.coco import COCO

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        pdb.set_trace()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
@torch.inference_mode()
def plot_demo(model, data_loader, device, save_dir):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    json_file = '/home/kwang/big_space/datasets/coco/annotations/instances_val2017.json'
    dataset_dir = '/home/kwang/big_space/datasets/coco/val2017/'

    coco = COCO(json_file)

    class_id = coco.cats
    for i, (images, targets) in enumerate(data_loader):
        try:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            img_id = targets[0]['image_id']
            image_info = coco.loadImgs(img_id)[0]

            image = cv2.imread(os.path.join(dataset_dir, image_info['file_name']))

            boxes = outputs[0]['boxes'].detach().cpu().numpy().astype(np.int32)
            labels = outputs[0]['labels'].detach().cpu().numpy()

            # Declaration: torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor
            boxes = torch.tensor(boxes, dtype=torch.float32)

            # 得到的是保留的索引
            keep = torchvision.ops.nms(boxes, outputs[0]['scores'], 0.5)

            boxes = boxes[keep]
            labels = labels[keep]

            boxes = boxes.tolist()
            labels = labels.tolist()

            if isinstance(labels, int):
                labels = [labels]
                boxes = [boxes]

            for box, label in zip(boxes,  labels):

                x, y, w, h = box

                if label not in list(range(1, 31)):
                    continue
                # pdb.set_trace()
                # v2.error: OpenCV(4.8.0) :-1: error: (-5:Bad argument) in function 'rectangle'
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
                anno_image_final = cv2.putText(anno_image, '{}'.format(class_id[label]['name']), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                image = anno_image_final
            # pdb.set_trace()
            # 保存为pdf
            # cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(str(i))), image)

            # 读取jpg，转化为pdf
            # image = cv2.imread(os.path.join(save_dir, '{}.jpg'.format(str(i))))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(16, 16))
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, '{}.pdf'.format(str(i))), bbox_inches='tight', pad_inches=0)
            plt.close()
            # pdb.set_trace()
            if i > 100:
                break
        except:
            continue

    # # only need four images
    # images = images
    # targets = targets
    #
    # images = list(img.to(device) for img in images)
    # outputs = model(images)
    # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    #
    # label_name = data_loader.dataset
    #
    # # plot the images with boxes and class name
    # for i, (image, target) in enumerate(zip(images, outputs)):
    #     boxes = target['boxes'].detach().cpu().numpy().astype(np.int32)
    #     labels = target['labels'].detach().cpu().numpy()
    #     image = image.permute(1,2,0).cpu().numpy()
    #     fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    #     for box, label in zip(boxes, labels):
    #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #         cv2.putText(image, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     ax.imshow(image)
    #     plt.savefig(os.path.join(save_dir, f"demo_{i}.png"))
    #     plt.close()





