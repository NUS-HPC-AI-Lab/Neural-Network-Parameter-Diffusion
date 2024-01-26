import abc
import hydra

class BaseTask():
    def __init__(self, config, **kwargs):
        self.cfg = config
        self.task_data = self.init_task_data()

    def init_task_data(self):
        pass

    def build_model(self):
        # get from param_data or init from cfg
        return hydra.utils.instantiate(self.cfg.model)

    def build_optimizer(self, net):
        # get from param_data or init from cfg
        return hydra.utils.instantiate(self.cfg.optimizer, net.parameters())

    def get_task_data(self):
        return self.task_data

    def get_param_data(self):
        return self.param_data

    @property
    def param_data(self):
        return self.set_param_data()

    @abc.abstractmethod
    def set_param_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_g_model(self, **kwargs):
        # test generation model
        pass

    @abc.abstractmethod
    def train_for_data(self):
        raise NotImplementedError