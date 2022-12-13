import torch
from matplotlib import pyplot as plt
from .gru import GRUNet
from torch import nn


def train_gru(data, input_weights, hidden_weights, hidden_size=100,
              start_idx=0, end_idx=1000, verbose=False, reg=0.01, targets=None, washout=0):
    train_data = torch.tensor(data[start_idx:end_idx]).reshape(1, -1, 1).float()
    if targets is None:
        targets = torch.tensor(data[start_idx + 1:end_idx + 1]).reshape(-1, 1).float()[washout:]
    else:
        targets = torch.tensor(targets[start_idx:end_idx]).reshape(-1, 1).float()[washout:]
    model = GRUNet(1, hidden_size, 1, washout=washout)
    model.train()
    # A bit dirty: Set the reservoir weights of the GRU layer to assure they are the same
    model.gru.weight_ih_l0 = nn.Parameter(input_weights.reshape(3 * hidden_size, -1), requires_grad=True)
    model.gru.weight_hh_l0 = nn.Parameter(hidden_weights.reshape(3 * hidden_size, -1), requires_grad=True)
    h = model.init_hidden(1).float()

    hidden_states, h_new = model.gru(train_data, h)
    hidden_states = hidden_states.reshape(-1, hidden_size)[washout:]
    output_weights = torch.linalg.solve(torch.mm(hidden_states.T, hidden_states) + reg * torch.eye(hidden_size),
                                        torch.mm(hidden_states.T, targets))
    model.w_out = output_weights
    prediction_tr, _ = model(train_data, h)
    training_loss = torch.nn.functional.mse_loss(prediction_tr, targets)
    if verbose:
        print("GRU Training complete. Loss:", training_loss.item())
    return model, hidden_states, h_new, training_loss.item()


def validate_gru(y, input_weights, hidden_weights, hidden_size=100,
                 start_idx=0, end_idx=1000, plot=False, targets=None, washout=0):
    train_end_idx = int((end_idx-1)/2)
    test_start_idx = train_end_idx + 1
    model, _, last_hidden, train_loss = train_gru(y, input_weights, hidden_weights, hidden_size,
                                                  start_idx=start_idx, end_idx=train_end_idx, targets=targets)
    test_data = torch.tensor(y[test_start_idx:end_idx-1]).reshape(1, -1, 1).float()
    if targets is None:
      test_targets = torch.tensor(y[test_start_idx+1:end_idx]).float().reshape(-1, 1)[washout:]
    else:
      test_targets = torch.tensor(targets[test_start_idx:end_idx-1]).float().reshape(-1, 1)[washout:]
    model.eval()
    prediction_te, h = model(test_data, last_hidden)
    mse = torch.nn.functional.mse_loss(prediction_te, test_targets)
    #print("MSE: ", mse)
    if plot:
        plt.figure(figsize=(25,4))
        plt.plot(test_targets.detach().numpy(), label="Targets Test")
        plt.plot(prediction_te.detach().numpy(), label="Prediction Test")
        plt.title("GRU")
        plt.legend()
        plt.show()
    return train_loss, mse.item()
