import torch
from torch import nn
import numpy as np


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, washout=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.washout = washout
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bias=True)
        self.w_out = torch.rand(hidden_dim, output_dim)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = torch.mm(out.reshape(-1, self.hidden_dim)[self.washout:], self.w_out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(1, batch_size, self.hidden_dim).zero_()
        return hidden


def generate_series(output_weight, length, reservoir):
    hidden_weights, input_weights = reservoir
    hidden_dim = output_weight.shape[0]
    model = GRUNet(1, hidden_dim, 1)
    model.gru.weight_ih_l0 = nn.Parameter(input_weights.reshape(3 * hidden_dim, -1), requires_grad=True)
    model.gru.weight_hh_l0 = nn.Parameter(hidden_weights.reshape(3 * hidden_dim, -1), requires_grad=True)
    h = model.init_hidden(1)
    y = 0.
    time_series = [y]
    for i in range(length):
        out, h = model(torch.tensor(y).reshape(1,-1,1).float(), h)
        new_y = out.reshape(-1).item()
        time_series.append(new_y)
        y = new_y
    return np.array(time_series)


# A cycle reservoir just like the one for ESNs
def cycle_reservoir_construction(us, vs, reservoir_size=100):
    hw = torch.zeros((3, reservoir_size, reservoir_size))
    iw = torch.zeros((3, reservoir_size, 1))
    for i in range(3):
        idxs = np.arange(0, reservoir_size-1)
        hw[i, idxs+1, idxs] = us[i]
        hw[i, 0, reservoir_size-1] = us[i]
        # Random Signed Input weights
        c = torch.randint(0,10,(reservoir_size, 1))
        v = torch.full((reservoir_size, 1), vs[i])
        iw[i] = torch.where(c % 2 == 0, v, v*-1)
    return hw, iw


def default_reservoir(reservoir_size=100):
    return cycle_reservoir_construction([0.9, 0.8, 0.7], [0.02, 0.03, 0.04], reservoir_size)


# Trained with hillclimbing algorithm
def kepler_reservoir(reservoir_size=100):
    return cycle_reservoir_construction([0.03478323, 1.14405769, 1.47451068], [0.00801442, 0.60542253, 0.11416187],
                                        reservoir_size)


def cauchy_reservoir(reservoir_size=100):
    return cycle_reservoir_construction([0.69069763, 0.81299484, 0.85773752], [0.60836169, 0.88066761, 1.2982216],
                                        reservoir_size)
