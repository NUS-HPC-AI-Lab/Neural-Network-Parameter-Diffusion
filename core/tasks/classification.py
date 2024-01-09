import hydra.utils

from .base_task import  BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import datetime
from core.utils import *
import glob
import omegaconf
import json


class CFTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(CFTask, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()

    def init_task_data(self):
        return VisionData(self.cfg.data)

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
        params_num = torch.squeeze(param).shape[0]  # + 30720
        assert (target_num == params_num)
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                target = target.to(torch.int64)
                test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

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
        net = self.build_model()
        optimizer = self.build_optimizer(net)
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
        epoch = self.cfg.epoch
        save_num = self.cfg.save_num_model
        all_epoch = epoch + save_num

        best_acc = 0
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        train_layer = self.cfg.train_layer

        data_path = getattr(self.cfg, 'save_root', 'param_data')

        tmp_path = os.path.join(data_path, 'tmp')
        final_path = os.path.join(data_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)


        save_model_accs = []
        parameters = []

        net = net.cuda()
        for i in range(0, all_epoch):
            self.train(net, criterion, optimizer, train_loader, i)
            acc = self.test(net, criterion, eval_loader)
            best_acc = max(acc, best_acc)

            if i == (epoch - 1):
                print("saving the model")
                torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
                fix_partial_model(train_layer, net)
                parameters = []
            if i >= epoch:
                parameters.append(state_part(train_layer, net))
                save_model_accs.append(acc)
                if len(parameters) == 10 or epoch == all_epoch - 1:
                    torch.save(parameters, os.path.join(tmp_path, "p_data_{}.pt".format(epoch)))
                    parameters = []

            scheduler.step()
        print("training over")

        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'performance': save_model_accs,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_accs

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

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


    def build_model(self):
        # get from param_data or init from cfg
        return hydra.utils.instantiate(self.cfg.model)


    def build_optimizer(self, net):
        # get from param_data or init from cfg
        return hydra.utils.instantiate(self.cfg.optimizer, net.parameters())



