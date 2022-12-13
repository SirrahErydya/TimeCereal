import torch.nn as nn
import torch
import numpy as np


class MyESN(nn.Module):
    def __init__(self, time_series, res_size, u, v, leaking_rate=1., train_hidden=False, washout=250):
        super(MyESN, self).__init__()
        self.input = time_series
        self.input_dim = time_series.shape[1]
        self.sequence_length = time_series.shape[0]
        self.output_dim = self.input_dim
        self.res_size = res_size
        self.u = nn.Parameter(u, requires_grad=train_hidden)
        self.v = nn.Parameter(v, requires_grad=train_hidden)
        self.leaking_rate = leaking_rate
        # self.output_layer = nn.Linear(self.res_size, self.output_dim, bias=False)
        self.w = torch.rand(self.res_size + 1, self.output_dim)
        self.washout = washout
        self.hidden_states = self.fit(self.input, torch.zeros((self.res_size, self.input_dim)))

    def fit(self, sequence, x):
        T = sequence.shape[0]
        X = torch.zeros((T, self.res_size + 1, self.input_dim))
        for t in range(T):
            y = sequence[t]
            x = (1 - self.leaking_rate) * x + torch.tanh(torch.mm(self.u.data, x) + (self.v.data * y))
            X[t][:-1] = x
            X[t][-1] = 1
        return X[self.washout:]

    def forward(self, time_series, x=None):
        if self.training:
            assert time_series.shape == self.input.shape
            X = self.hidden_states.reshape(self.sequence_length - self.washout, -1)
        else:
            assert x is not None
            X = self.fit(time_series, x).reshape(time_series.size()[0] - self.washout, -1)
        return torch.mm(X, self.w)


def cycle_reservoir_construction(u, v, reservoir_size=100):
    hw = torch.zeros((reservoir_size, reservoir_size))
    idxs = np.arange(0, reservoir_size-1)
    hw[idxs+1, idxs] = u
    hw[ 0, reservoir_size-1] = u
    # Random Signed Input weights
    c = torch.randint(0,10,(reservoir_size, 1))
    v = torch.full((reservoir_size, 1), v)
    iw = torch.where(c % 2 == 0, v, v*-1)
    return hw, iw


# A default reservoir for test purposes
def default_reservoir():
    return cycle_reservoir_construction(0.9414, 0.0071)