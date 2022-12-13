import os.path

from gp import training as gp_training
from gru import gru
from data import data_loading
from gru import training as gru_training
from autoencoder.vanilla_ae import AutoEncoder
from autoencoder import training as aetraining
import torch
import numpy as np
import pickle
from visualisation import visualisation as vis
from tqdm import tqdm
import sys
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULT_PTH = "../RESULTS"


def get_rnn(time_series, hidden_weights, input_weights,
            rnn_type='gru', gaussian_kernel='matern', gp_epochs=100, sample_size=1000):
    """
    :param id: Identificator of the time series
    :param time_series: A tuple of times and according values
    :param rnn_type: Type of the RNN. can be GRU or ESN
    :param gaussian_kernel: Type of Gaussian kernel
    :param gp_epochs: Number of training epochs for the gaussian process
    :param sample_size: How many samples should belong to the simulated light curve
    :param plot: Make plots and save them
    :return: A recurrent network able to model the input time series
    """
    timestamps = time_series.property("timestamps")
    values = time_series.property("values")
    errors = time_series.property("errors")
    if gaussian_kernel is not None:
        std = values.std()
        values /= std
        errors /= std
        test_times = torch.linspace(np.min(timestamps), np.max(timestamps), sample_size)
        if hasattr(time_series, "gp"):
            rnn_input = time_series.gp
        else:
            test_times, prediction, _, _ = gp_training.make_gp_simulation(time_series, DEVICE, gaussian_kernel,
                                                                          gp_epochs, sample_size)
            test_times = test_times.cpu().numpy()
            rnn_input = prediction.mean.cpu().detach().numpy()

    else:
        rnn_input = values

    # Train an RNN
    print("Training the RNN...")
    hidden_size = hidden_weights.size()[1]
    if rnn_type == 'gru':
        rnn, hidden_state, last_hidden,loss = gru_training.train_gru(rnn_input, input_weights, hidden_weights,
                                                                     hidden_size, start_idx=0, end_idx=sample_size-1)
    else:
        raise ValueError("RNNs of type {0} are not supported.".format(rnn_type))
    print("Training loss: {0}".format(loss))
    # Predict with the RNN
    print("Predicting with the RNN...")
    test_tensor = torch.tensor(rnn_input).reshape((1, -1, 1)).float()
    rnn.eval()
    h = rnn.init_hidden(1).float()
    rnn_prediction, _ = rnn(test_tensor, h)
    rnn_observation = rnn_prediction.reshape(-1)

    print("Save all properties.")
    if gaussian_kernel is None:
        test_times = np.arange(len(rnn_observation))
    time_series.set_property("rnn_prediction", rnn_observation.detach().cpu().numpy())
    time_series.set_property("sample_timestamps", test_times)
    torch.cuda.empty_cache()
    time_series.set_property("rnn_x", hidden_state.detach().cpu().numpy())
    time_series.set_property("rnn_w", rnn.w_out.reshape(-1).detach().cpu().numpy())
    time_series.set_property("rnn_y", test_tensor.reshape(-1)[1:].detach().cpu().numpy())
    if gaussian_kernel is not None:
        time_series.set_property("gp", rnn_input)
    print("All done :)")


def train_output_weights(dataset, reservoir, rnn_type='gru', gaussian_kernel='matern',
                         gp_epochs=100, sample_size=1000):
    hidden_weights, input_weights = reservoir
    for event in tqdm(list(dataset)):
        ts_id = event.id
        get_rnn(event, hidden_weights, input_weights,  rnn_type=rnn_type,
                gaussian_kernel=gaussian_kernel, gp_epochs=gp_epochs, sample_size=sample_size)
        with open(os.path.join(event_path, "{0}.pk").format(ts_id), "wb") as cc:
            pickle.dump(event, cc)


def get_2d_projection(cc_events, events_path, ae_path, reservoir_size):
    event_path = os.path.join(events_path, "{0}.pk")
    ds_points = [(torch.tensor(event.rnn_w),
                  torch.tensor(event.rnn_x),
                  torch.tensor(event.rnn_y)) for event in cc_events]
    time_series_dataset = aetraining.TimeseriesDataset(ds_points)
    ae, train_loss = aetraining.train_autoencoder(time_series_dataset, reservoir_size, DEVICE)
    print("Average training loss: {0}".format(train_loss))
    ae = ae.cpu()
    for event in cc_events:
        w_2d = ae.encoder(torch.tensor(event.rnn_w).reshape(1,-1))
        ae_recon = np.matmul(event.rnn_x.reshape(-1, len(event.rnn_w)),
                             ae.decoder(w_2d).reshape(-1,1).detach().numpy()).reshape(-1)
        event.set_property("w_2d", w_2d.reshape(-1).detach().numpy())
        event.set_property("ae_y", ae_recon.reshape(-1))

        with open(event_path.format(event.id), "wb") as wpth:
            pickle.dump(event, wpth)
    torch.save(ae.state_dict(), ae_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--name', dest='dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--train', dest='train', type=bool, help='If true, train the models', default=False)
    parser.add_argument('--train_ae', dest='train_ae', type=bool, help='If true, train Autoencoder', default=False)
    parser.add_argument('--sparsen', dest='sparsen', type=bool, help='Sparsen the input data', default=False)
    parser.add_argument('--path', dest='path', help='Working path', default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Device:", DEVICE)
    args = parse_arguments()
    working_path = os.path.join(RESULT_PTH, args.dataset_name, args.path)
    if not os.path.exists(working_path):
        os.mkdir(working_path)
    event_path = os.path.join(working_path, "CCEvents")
    ae_path = os.path.join(working_path, "autoencoder.mdl")

    reservoir_size = 100

    data = data_loading.load_data(args.dataset_name, event_path, sparsen=args.sparsen)
    if args.train:
        kernel = "matern"
        sample_size = 1000
        is_labeled = False
        reservoir = gru.kepler_reservoir(reservoir_size)
        if args.dataset_name == "cauchy" or args.dataset_name == "coffee":
            reservoir = gru.default_reservoir(reservoir_size)
            kernel = None
            is_labeled = True
            sample_size = data[0].length
        train_output_weights(data, reservoir, sample_size=sample_size, gaussian_kernel=kernel)
    if args.train or args.train_ae:
        get_2d_projection(data, event_path, ae_path, reservoir_size)

    ae = AutoEncoder(reservoir_size)
    ae.load_state_dict(torch.load(ae_path))
    vis.scatter_plot(args.dataset_name, data, event_path, working_path, ae)

