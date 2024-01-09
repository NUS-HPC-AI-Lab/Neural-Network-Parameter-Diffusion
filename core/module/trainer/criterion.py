
import torch.nn as nn

class mse(nn.MSELoss):
    def __init__(self):
        super(mse, self).__init__()