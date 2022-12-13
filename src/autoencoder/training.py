import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from autoencoder.vanilla_ae import AutoEncoder
from matplotlib import pyplot as plt
import os
import sys
import pickle


class TimeseriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def ae_loss(prediction, ground_truth, mu, sigma):
    mse = F.mse_loss(prediction, ground_truth)
    kld = -0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp())
    return mse + kld


# define the sparse loss function
def sparse_loss(model, w_in):
    loss = 0
    values = w_in
    model_children = list(model.children())
    for i in range(len(model_children)):
        values = F.relu((model_children[i](values)))
        loss += torch.mean(torch.abs(values).cpu())
    return loss


def train_autoencoder(vector_dataset, vector_length, device, encoder_type="vanilla", reg=1e-06,
                      epochs=2000, learning_rate=1e-3, plot=False):
    train_loader = DataLoader(vector_dataset, batch_size=len(vector_dataset), shuffle=True)

    if encoder_type == "vanilla":
        ae = AutoEncoder(vector_length).to(device)
    else:
        raise ValueError("No such encoder type:", encoder_type)
    ae.train()
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)
    train_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        # Training
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_loader):
            w, X, y = data
            w_rec = ae(w.to(device)).cpu()
            y_rec = torch.zeros(y.shape)
            for i in range(y_rec.shape[0]):
                y_rec[i] = torch.mm(X[i].reshape(-1, vector_length), w_rec[i].reshape(-1,1)).reshape(-1)
            loss = F.mse_loss(y_rec, y) + reg * torch.sum(w_rec.to(device)**2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    # Plot
    if plot:
        plt.figure(figsize=(9,4))
        plt.plot(train_losses, label="Training loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Autoencoder training")
        plt.show()

        plt.figure(figsize=(9, 4))
        plt.plot(y[0].cpu().detach().numpy(), label="RNN output")
        plt.plot(y_rec[0].cpu().detach().numpy(), label="RNN reconstruction")
        plt.legend()
        plt.show()

    return ae, train_losses[-1]


def validate_ae(vector_dataset, vector_length, device):
    train_length = int(0.7 * len(vector_dataset))
    validation_length = len(vector_dataset) - train_length
    train_set, validation_set = torch.utils.data.random_split(vector_dataset, (train_length, validation_length))
    val_loader = DataLoader(vector_dataset, batch_size=len(validation_set), shuffle=True)

    ae, train_loss = train_autoencoder(train_set, vector_length, device)
    ae.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            w, X, y = data
            w_rec = ae(w.to(device)).cpu()
            y_rec = torch.zeros(y.shape)
            for i in range(y_rec.shape[0]):
                y_rec[i] = torch.mm(X[i].reshape(-1, vector_length), w_rec[i].reshape(-1, 1)).reshape(-1)
            loss = F.mse_loss(y_rec, y)
            val_losses.append(loss.item())
    return train_loss, val_losses[-1]


if __name__ == "__main__":
    events_path = sys.argv[1]
    cc_events = []
    for file in os.scandir(events_path):
        if file.is_file():
            with open(file, "rb") as cc:
                cc_event = pickle.load(cc)
                cc_events.append(cc_event)
    ds_points = [(torch.tensor(event.rnn_w.cpu().detach().numpy()),
                  torch.tensor(event.rnn_x.cpu().detach().numpy()),
                  torch.tensor(event.rnn_y.cpu().detach().numpy())) for event in cc_events]
    time_series_dataset = TimeseriesDataset(ds_points)
    train_loss, val_loss = validate_ae(ds_points, 100, "cpu")
    print("Training loss: {0}".format(train_loss))
    print("Test loss: {0}".format(val_loss))