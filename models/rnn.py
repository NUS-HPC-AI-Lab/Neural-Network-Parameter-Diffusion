# pytorch code from robert yang:
# https://github.com/neurogym/ngym_usage/tree/master/yang19
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import scipy as scipy
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

class CTRNN(nn.Module):

    """Continuous-time RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        mask: N_h x N_h mask either 2d or 1d
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, mask = None, **kwargs):

        super().__init__()
        self.input_size = input_size


        self.hidden_size = hidden_size
        self.tau = 100
        self.mask = mask

        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

        #initialize hidden to hidden weight matrix using the mask
        if mask is None:
            temp = 0
        else:
            self.h2h.weight.data = self.h2h.weight.data*torch.nn.Parameter(mask)

    def reset_parameters(self):
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden)
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)
        return output, hidden



class RNNNet(nn.Module):

    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, output_size, mask, **kwargs):
        super().__init__()

        #Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, mask = mask, **kwargs)
        #readout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #hidden dynamics
        rnn_activity, _ = self.rnn(x)
        #readout
        out = self.fc(rnn_activity)
        # return out, rnn_activity
        return out
    

def get_performance(net, env, num_trial=1000, device='cpu'):
    perf = 0

    for i in range(num_trial):

        env.new_trial()
        #Observations and ground truth
        ob, gt = env.ob, env.gt

        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.DoubleTensor).to(device)
        
        action_pred, _ = net(inputs)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        #Check if network prediction == ground truth
        perf += (gt[-1] == action_pred[-1, 0])

    perf /= num_trial
    return perf

def get_performance_proper(net, env, num_trial=1000, device='cpu'):
  perf = 0

  for i in range(num_trial):

      env.new_trial()
      #Observations and ground truth
      ob, gt = env.ob, env.gt

      ob = ob[:, np.newaxis, :]  # Add batch axis
      inputs = torch.from_numpy(ob).type(torch.DoubleTensor).to(device)

      action_pred, _ = net(inputs)
      action_pred = action_pred.detach().cpu().numpy()
      action_pred = np.argmax(action_pred, axis=-1)

      #Check if network prediction == ground truth
      perf += (gt[-1] == action_pred[-1, 0])

  perf /= num_trial
  return perf