import abc

class BasePreLayer(object):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def pre_process(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self, batch):
        raise NotImplementedError