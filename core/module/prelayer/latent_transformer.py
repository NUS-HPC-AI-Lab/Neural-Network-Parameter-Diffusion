from .base import BasePreLayer


class Param2Latent(BasePreLayer):
    def __init__(self, system_cls, checkpoint_path=None, **kwargs):
        super(Param2Latent, self).__init__(**kwargs)
        if checkpoint_path is not None:
            self.system = system_cls.load_from_checkpoint(checkpoint_path)
        else:
            self.system = system_cls(**kwargs)

    def pre_process(self, batch):
        return self.system.encode(batch)

    def post_process(self, batch):
        return self.system.decode(batch)