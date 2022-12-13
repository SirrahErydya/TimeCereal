from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=length, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=length),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
