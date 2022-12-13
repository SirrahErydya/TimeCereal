from gp import training as gptraining
from gru import training as grutraining
from gru.gru import default_reservoir, kepler_reservoir
from autoencoder import training as aetraining
from data import data_loading
import sys
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    events = data_loading.load_data(dataset_name, None, False)

    with_gp = dataset_name != "cauchy" and dataset_name != "coffee"
    hidden_weights, input_weights = default_reservoir()
    sample_size = events[0].length
    hidden_dim = 100

    print("Test GRU and GP...")
    gp_tr_losses, gp_te_losses = [], []
    gru_tr_losses, gru_te_losses = [], []
    datapoints = []
    plt.figure(figsize=(9, 4))
    for event in tqdm(events):
        rnn_input = event.values
        if with_gp:
            std = event.values.std()
            event.values /= std
            event.errors /= std

            hidden_weights, input_weights = kepler_reservoir()
            test_times, rnn_input, _,_ = gptraining.make_gp_simulation(event, DEVICE, "matern", 100, sample_size,
                                                                       plot=True)
            rnn_input = rnn_input.mean.cpu().detach().numpy()
            event.set_property("sample_timestamps", test_times)
            event.set_property("gp", rnn_input)
            _, _, train_loss, test_loss = gptraining.make_gp_simulation(event, DEVICE, "matern", 100, sample_size,
                                                                        validate=True, plot=False)
            gp_tr_losses.append(train_loss)
            gp_te_losses.append(test_loss)
        rnn, hidden_state, last_hidden, loss = grutraining.train_gru(rnn_input, input_weights, hidden_weights,
                                                                     hidden_dim, start_idx=0, end_idx=sample_size - 1)
        event.set_property("rnn_x", hidden_state.cpu().detach().numpy())
        event.set_property("rnn_w", rnn.w_out.reshape(-1).cpu().detach().numpy())
        event.set_property("rnn_y", torch.tensor(rnn_input).reshape(-1)[1:].cpu().detach().numpy())
        datapoints.append((torch.tensor(event.rnn_w).float(),
                           torch.tensor(event.rnn_x).float(),
                           torch.tensor(event.rnn_y).float()))
        train_loss, val_loss = grutraining.validate_gru(rnn_input, input_weights, hidden_weights, hidden_dim,
                                                        start_idx=0, end_idx=sample_size - 1)
        gru_tr_losses.append(train_loss)
        gru_te_losses.append(val_loss)
    plt.show()
    gp_tr_loss = np.average(gp_tr_losses)
    gp_te_loss = np.average(gp_te_losses)
    gru_tr_loss = np.average(gru_tr_losses)
    gru_te_loss = np.average(gru_te_losses)

    print("Test Autoencoder...")
    time_series_dataset = aetraining.TimeseriesDataset(datapoints)
    ae_tr_loss, ae_te_loss = aetraining.validate_ae(datapoints, hidden_dim, DEVICE)

    print("###################################################")
    print("All done! Here are the losses")
    if with_gp:
        print("###################################################")
        print("# GP")
        print("Training Average: ", gp_tr_loss)
        print("Training Average: ", gp_te_loss)
    print("###################################################")
    print("# GRU")
    print("Training Average: ", gru_tr_loss)
    print("Validation Average", gru_te_loss)
    print("###################################################")
    print("# Autoencoder")
    print("Training Average: ", ae_tr_loss)
    print("Validation Average", ae_te_loss)
    print("###################################################")
