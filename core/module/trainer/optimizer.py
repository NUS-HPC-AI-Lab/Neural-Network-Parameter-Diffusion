from torch.optim import SGD, Adam, AdamW, RMSprop, Adadelta, Adagrad, Adamax, ASGD, LBFGS, Rprop


class adamw(AdamW):
    def __int__(self, **kwargs):
        super(adamw, self).__init__(**kwargs)