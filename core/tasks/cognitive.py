import pdb

import hydra.utils

from .base_task import BaseTask
from core.data.cognitive_dataset import CognitiveData
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import datetime
from core.utils import *
import glob
import omegaconf
import json
import pandas as pd

from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from neurogym.utils.plotting import *
from Mod_Cog.mod_cog_tasks import *


class CogTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(CogTask, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataset()
        self.eval_loader = self.task_data.val_dataset()
        self.test_loader = self.task_data.test_dataset()

    def init_task_data(self):
        return CognitiveData(self.cfg.data)

    # override the abstract method in base_task.py
    def set_param_data(self):
        param_data = PData(self.cfg.param)
        self.model = param_data.get_model()
        self.train_layer = param_data.get_train_layer()
        return param_data
    

    def test_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0] 
        assert (target_num == params_num)
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(
            param, net, train_layer).to(param.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                #TODO: dataype?
                # target = target.to(torch.float)
                # sum up batch loss
                test_loss += F.cross_entropy(output,
                                             target, size_average=False).item()

                total += data.shape[0]
                pred = torch.max(output, 1)[1]
                output_list += pred.cpu().numpy().tolist()
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list



    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0]  # + 30720
        assert (target_num == params_num)
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(
            param, net, train_layer).to(param.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                test_loss += F.cross_entropy(output,
                                             target, size_average=False).item()

                total += data.shape[0]
                pred = torch.max(output, 1)[1]
                output_list += pred.cpu().numpy().tolist()
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list


    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):
        ckpt_path = self.cfg.param.ckpt_path
        final_path = self.cfg.param.final_path
        train_layer = self.cfg.train_layer

        pdata = []
        save_model_accs = []
        for seed in self.cfg.param.seeds:
            files = glob.glob(os.path.join(ckpt_path, f'hidden_32_seed_{seed}_epoch_*.pt'))
            files = sorted(files, key=extract_epoch)    #sort by the corresponding epoch
            for file in files:
                buffer = torch.load(file)
                param = []
                for key in buffer.keys():
                    if train_layer == 'all': 
                        param.append(buffer[key].data.reshape(-1))
                    elif key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
            
            log = os.path.join(self.cfg.param.perf_path, f'hidden_32_seed_{seed}_eval.csv')
            save_model_accs.append(pd.read_csv(log)['perf'].to_list())

        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(ckpt_path)
        print(f"path {ckpt_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(ckpt_path, file)),
            'train_layer': train_layer,
            'performance': save_model_accs,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': list(save_model_accs)

        }
        print(np.shape(batch))
        print(np.shape(save_model_accs))
        json.dump(json_state, open(
            os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        # shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, criterion, optimizer, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(self, net, criterion, testloader):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            return 100. * correct / total
