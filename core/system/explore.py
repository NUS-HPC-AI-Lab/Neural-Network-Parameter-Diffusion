import pdb
from typing import Any
import torch
from .ae_ddpm import AE_DDPM
import numpy as np

class Explore(AE_DDPM):
    def __init__(self, config):
        super(Explore, self).__init__(config)


    def test_step(self, batch, batch_idx, **kwargs: Any):
        # self.train_and_val_exp(batch)
        self.train_and_val_exp(batch)

    def train_and_val_exp(self, batch):
        batch = self.pre_process(batch)
        acc_list = []
        train_acc_list = []
        # pdb.set_trace()
        param_list = []
        while len(acc_list) < 100:
            outputs = self.generate(batch, 100)
            params = self.post_process(outputs)

            for param in params:
                acc, test_loss, output_list = self.task_func(param)
                print("generating acc: {}".format(acc))
                if acc > 92:
                    acc_list.append(acc)
                    param_list.append(param)

        for param in param_list:
            train_acc, train_loss, output_list = self.task.val_g_model(param)
            print("train acc: {}".format(train_acc))
            train_acc_list.append(train_acc)
        acc_result = {
            'acc': acc_list,
            'train_acc': train_acc_list,
        }
        torch.save(acc_result, '/home/kwang/zhouyukun/outputs/nndiff/acc_result_t100.pt')

    def train_further(self, batch):
        result = []
        max_iterations = 500
        batch = self.pre_process(batch)

        num = 0
        model = self.model.ema
        model.eval()
        noise_fn = torch.randn
        while num < 10:
            output = self.generate(batch, 1)
            param = self.post_process(output)
            acc, test_loss, output_list = self.task_func(param)
            shape = (1, 1, batch.shape[1] * batch.shape[2])
            if acc < 94.0 and acc > 90:
                end_t = 0
                iteration_num = 0
                tmp_acc = acc
                print("{} iteration num, acc {}".format(iteration_num, tmp_acc))
                # pdb.set_trace()
                while tmp_acc < 94.0 and iteration_num < max_iterations:
                    output, pred_x0 = self.p_sample(
                        model=model,
                        x=output,
                        t = torch.full((shape[0],), end_t, dtype=torch.int64).to(batch.device),
                        noise_fn=noise_fn,
                        return_pred_x0=True
                    )
                    param = self.post_process(output)
                    tmp_acc, test_loss, output_list = self.task_func(param)
                    print("{} iteration num, acc {}".format(iteration_num, tmp_acc))
                    iteration_num += 1
            else:
                continue

    def cal_wrong_iou(self, pred1, pred2, target):
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        wrong_index1 = np.where(pred1 != target)[0]
        wrong_index2 = np.where(pred2 != target)[0]
        wrong_index1 = set(wrong_index1.tolist())
        wrong_index2 = set(wrong_index2.tolist())

        return len(wrong_index1.intersection(wrong_index2)) / len(wrong_index1.union(wrong_index2))

    def find_max_iou(self, pred1, pred_list, groudtruth):
        iou_array =  np.zeros((len(pred_list),))
        for i, pred in enumerate(pred_list):
            iou_array[i] = self.cal_wrong_iou(pred1, pred, groudtruth)
        return iou_array


    def original_sample_higher_iou(self, batch):
        max_iteration = 1000
        iteration_num = 0
        original_output_list = []

        good_param = batch
        input_accs = []
        for i, param in enumerate(good_param):
            acc, test_loss, output_list = self.task_func(param)
            input_accs.append(acc)
            original_output_list.append(output_list)


        # cal iou matrix
        groudtruth = self.task.test_loader.dataset.targets
        batch = self.pre_process(batch)
        while iteration_num < max_iteration:
            output = self.generate(batch, 1)
            param = self.post_process(output)
            acc, test_loss, output_list = self.task_func(param)

            if acc > 99.2:
                iou_array = self.find_max_iou(output_list, original_output_list, self.task.test_loader.dataset.targets)
                print("max iou: {}".format(np.max(iou_array).max()))
                pdb.set_trace()
            print(acc)
            iteration_num += 1

