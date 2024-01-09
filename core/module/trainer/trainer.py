from pytorch_lightning import Trainer


class trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
