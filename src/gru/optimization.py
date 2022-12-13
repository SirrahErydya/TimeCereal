from gru import gru, training
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import json
import os
from matplotlib import pyplot as plt
from data import data_loading
import numpy as np

RESULT_PTH = "/home/kollasfa/MasterThesis/RESULTS"


class OptimizationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float()

    def __len__(self):
        return len(self.data)


def optimize(dataset, reservoir_size=100):
    init_us = nn.Parameter(torch.rand(3), requires_grad=True)
    init_vs = nn.Parameter(torch.rand(3), requires_grad=True)
    print("Initial parameters: u - {0}; v - {1}".format(init_us, init_vs))
    init_hw, init_iw = gru.cycle_reservoir_construction(init_us, init_vs)

    train_length = int(0.8 * len(dataset))
    validation_length = len(dataset) - train_length
    train_set, validation_set = torch.utils.data.random_split(dataset, (train_length, validation_length))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1)

    optimizer = torch.optim.Adam([init_us, init_vs], lr=0.01)
    optimizer.zero_grad()
    losses = []
    for data in tqdm(train_loader):
        data = data.reshape(-1)
        rnn = gru.GRUNet(1, reservoir_size, 1, washout=0)
        rnn.gru.weight_ih_l0 = nn.Parameter(init_iw.reshape(3 * reservoir_size, -1), requires_grad=False)
        rnn.gru.weight_hh_l0 = nn.Parameter(init_hw.reshape(3 * reservoir_size, -1), requires_grad=False)
        loss = training.train_gru(data, rnn)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plt.figure(figsize=(10,10))
    plt.plot(losses)
    plt.show()
    return init_us, init_vs


def optimize_bruteforce(dataset, reservoir_size=100, epochs=1000):
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    best_us = None
    best_vs = None
    best_loss = torch.inf
    losses = []
    for epoch in range(epochs):
        print("Epoch {0}/{1}:".format(epoch+1, epochs))
        if epoch == 0:
            # Also try default reservoir
            us = [0.9, 0.8, 0.7]
            vs = [0.02, 0.03, 0.04]
        else:
            us = torch.rand(3)
            vs = torch.rand(3)

        hw, iw = gru.cycle_reservoir_construction(us, vs, reservoir_size=reservoir_size)
        print("Parameter Guess: u - {0}; v - {1}".format(us, vs))
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.reshape(-1)
            data = data.cpu().detach().numpy()
            size = data.shape[0]
            _, _, _, loss = training.train_gru(data, iw, hw, hidden_size=reservoir_size, start_idx=0, end_idx=size-1)
            total_loss += loss
        total_loss /= len(dataset)
        losses.append(total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            best_us = us
            best_vs = vs
        print(total_loss)
    plt.figure(figsize=(10,10))
    plt.plot(losses)
    plt.show()
    return best_us, best_vs, best_loss


def optimize_hillclimb(dataset, init_u, init_v, reservoir_size=100, epochs=1000):
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Start with default reservoir
    best_us = np.array(init_u)
    best_vs = np.array(init_v)

    # Helper function to calculate the average loss over the dataset with a reservoir
    def dataset_loss(us, vs):
        total_train_loss = 0
        total_test_loss = 0
        hw, iw = gru.cycle_reservoir_construction(us, vs, reservoir_size=reservoir_size)
        for data in tqdm(train_loader):
            data = data.reshape(-1)
            data = data.cpu().detach().numpy()
            train_loss, test_loss = training.validate_gru(data, iw, hw, hidden_size=reservoir_size,
                                                          start_idx=0, end_idx=len(dataset))
            total_train_loss += train_loss
            total_test_loss += test_loss
        total_train_loss /= len(dataset)
        total_test_loss /= len(dataset)
        return total_train_loss, total_test_loss

    final_train_loss, best_test_loss = dataset_loss(best_us, best_vs)

    train_losses, test_losses = [final_train_loss], [best_test_loss]
    for epoch in tqdm(range(epochs)):
        # Add 20% Gaussian Noise
        new_us = best_us + np.random.normal(0.0, 0.2 * np.std(best_us), 3)
        new_vs = best_vs + np.random.normal(0.0, 0.2 * np.std(best_vs), 3)
        print("Parameter Guess: u - {0}; v - {1}".format(new_us, new_vs))

        new_train_loss, new_test_loss = dataset_loss(new_us, new_vs)
        print("Train loss: {0}".format(new_train_loss))
        print("Test loss: {0}".format(new_test_loss))
        if new_test_loss <= best_test_loss:
            best_us = new_us
            best_vs = new_vs
            final_train_loss = new_train_loss
            best_test_loss = new_test_loss
        train_losses.append(final_train_loss)
        test_losses.append(best_test_loss)

    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.show()

    return best_us, best_vs, best_test_loss, final_train_loss


if __name__ == "__main__":
    ds_name = sys.argv[1]
    event_path = os.path.join(RESULT_PTH, ds_name, "CCEvents")
    events = data_loading.load_cc_events(event_path)
    values = [event.values for event in events]
    dataset = OptimizationDataset(values)
    # Try with results from Brute Force
    opt_us, opt_vs, opt_loss = optimize_bruteforce(dataset)
    #opt_us, opt_vs, opt_loss, t_loss = optimize_hillclimb(gp_dataset, [0.1790, 0.8733, 0.9557], [0.9177, 0.8614, 0.2245])
    print("Optimal parameters: u - {0}; v - {1}".format(opt_us, opt_vs))
    #print("Train Loss: {0}".format(t_loss))
    print("Test Loss: {0}".format(opt_loss))
