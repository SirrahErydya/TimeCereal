from tqdm import tqdm
import numpy as np
import gpytorch
from data.data_loading import load_kepler_curve, KEPLER_PATH
import os
from gp import gp
import torch
from matplotlib import pyplot as plt


def gaussian_training_loop(x_train, y_train, device, kernel="matern",
                           init_learning_rate=0.1, final_learning_rate=0.01,  epochs=150, noise=None, plot=False):
    if noise is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    else:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise).to(device)
    model = None
    if kernel == "poly":
        model = gp.PPolynomialGP(x_train, y_train, likelihood).to(device)
    elif kernel == "matern":
        model = gp.MaternGP(x_train, y_train, likelihood).to(device)
    elif kernel == "periodic":
        model = gp.PeriodicGP(x_train, y_train, likelihood).to(device)
    elif kernel == "rbf":
        model = gp.RBFGP(x_train, y_train, likelihood).to(device)
    else:
        raise ValueError("No GP with a kernel of type {0}.".format(kernel))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate, betas=(0.5, 0.99))
    gamma = (final_learning_rate / init_learning_rate) ** (1.0 / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    for i in tqdm(range(epochs)):
        # Set the gradients from previous iteration to zero
        optimizer.zero_grad()
        # Output from model
        output = model(x_train.to(device))
        # Compute loss and backprop gradients
        loss = -mll(output, y_train.to(device))
        losses.append(loss.item())
        loss.backward()
        # if i % 100 == 0:
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()
        scheduler.step()
    if plot:
        plt.plot(np.array(losses))
        plt.ylabel("Loss")
        #plt.show()
    return model, likelihood, losses[-1]


def make_gp_simulation(timeseries, device, gaussian_kernel, gp_epochs, sample_size, validate=False, plot=False):
    # Make tensors. The byteswap is necessary because apparently, FITS is weird...
    timestamps = timeseries.timestamps
    values = timeseries.values
    errors = timeseries.errors
    training_times = torch.tensor(timestamps.newbyteorder().byteswap()).float()
    training_targets = torch.tensor(values.newbyteorder().byteswap()).float()
    training_errors = torch.tensor(errors.newbyteorder().byteswap()).float()

    validation_targets = training_targets
    sim_times = torch.linspace(np.min(timestamps), np.max(timestamps), sample_size)

    if validate:
        validation_times = training_times[1::2]
        validation_targets = training_targets[1::2]
        training_times = training_times[::2]
        training_targets = training_targets[::2]
        training_errors = training_errors[::2]

    # Train a gaussian process
    print("Training a gaussian process...")
    # training_times = training_times.to(device)
    # training_targets = training_targets.to(device)
    gp, likelihood, train_loss = gaussian_training_loop(training_times, training_targets, device,
                                                        kernel=gaussian_kernel, epochs=gp_epochs, noise=training_errors,
                                                        plot=plot)

    # Make an evenly distributed prediction
    print("Predicting the Gaussian Observation...")
    gp.eval()
    prediction = gp(sim_times.to(device))
    # Calculate loss of the predicition
    test_loss = 0
    if validate:
        val_prediction = gp(validation_times.to(device))
        test_loss = torch.nn.functional.mse_loss(val_prediction.mean.cpu(), validation_targets).item()
    return sim_times, prediction, train_loss, test_loss


def validate_gp(events, device):
    train_losses = []
    val_losses = []
    for event in tqdm(events):
        test_times, prediction, train_loss, val_loss = make_gp_simulation(event, device, "matern", 100, 1000, validate=True)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return np.average(train_losses), np.average(val_losses)


if __name__ == "__main__":
    timeseries = load_kepler_curve(os.path.join(KEPLER_PATH, "0007/000757076/kplr000757076-2013131215648_llc.fits"))
    times, pred = make_gp_simulation(timeseries, "cpu", "matern", 100, 1000)
    obs = pred.mean.cpu().detach().numpy()
    plt.figure(figsize=(9, 4))
    plt.plot(timeseries[0], timeseries[1], label='Real Data')
    plt.plot(times, obs, label='Gaussian Simulation')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Flux (electrons/second)')
    plt.show()

