import sys
import os
from data import data_loading
from gru.optimization import optimize_hillclimb, OptimizationDataset

if __name__ == "__main__":
    event_path = sys.argv[1]
    events = data_loading.load_cc_events(event_path)
    if hasattr(events[0], "gp"):
        values = [event.gp for event in events]
        us = [0.1790, 0.8733, 0.9557]
        vs = [0.9177, 0.8614, 0.2245]
    else:
        values = [event.values for event in events]
        us = [0.7335, 0.8231, 0.9428]
        vs = [0.5431, 0.9657, 0.9323]
    dataset = OptimizationDataset(values)
    # Try with results from Brute Force
    opt_us, opt_vs, opt_loss, t_loss = optimize_hillclimb(dataset, us, vs)
    print("Optimal parameters: u - {0}; v - {1}".format(opt_us, opt_vs))
    print("Train Loss: {0}".format(t_loss))
    print("Test Loss: {0}".format(opt_loss))