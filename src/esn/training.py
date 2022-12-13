import torch
from esn.esn import MyESN


# We train the model by initially guessing the weights and solving the equation to get the output weights
def train_esn(data, input_weights, reservoir_weights, reservoir_size=300, start_idx=0, end_idx=5000, washout=0, targets=None):
    train_data = torch.tensor(data[start_idx:end_idx]).reshape((-1, 1)).float()
    if targets is not None:
        train_targets = torch.tensor(targets[start_idx:end_idx]).reshape((-1, 1)).float()[washout:]
    else:
        train_targets = torch.tensor(data[start_idx+1:end_idx+1]).reshape((-1, 1)).float()[washout:]

    model = MyESN(train_data, reservoir_size, reservoir_weights, input_weights, washout=washout)
    model.train()
    hidden_states = model.hidden_states.reshape(train_data.size()[0]-washout, -1)
    reg = 0.1
    output_weights = torch.linalg.solve(torch.mm(hidden_states.T,hidden_states) + reg*torch.eye(reservoir_size+1), torch.mm(hidden_states.T,train_targets))
    model.w = output_weights
    prediction = model(train_data)
    mse = torch.nn.functional.mse_loss(prediction, train_targets)
    return model, mse