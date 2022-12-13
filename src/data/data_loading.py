from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import csv
from coreconcepts.events import AstroEvent
import pickle


KEPLER_PATH = "../DATA/Kepler"
CAUCHY_PATH = "../DATA/Cauchy"
COFFEE_PATH = "../DATA/Coffee"
#KEPLER_PATH = "/home/fenja/Desktop/MASTERARBEIT/DATA/Kepler/Q17"


def make_cc_event(id, time_series, save_path, label=""):
    timestamps, values, errors = time_series
    ts_start, ts_end = timestamps[0], timestamps[-1]
    cc_event = AstroEvent(id, start=ts_start, end=ts_end, timestamps=timestamps, values=values, errors=errors,
                          length=len(timestamps), label=label)
    if save_path is not None:
        pth = save_path.format(id)
        with open(pth, "wb") as wpth:
            pickle.dump(cc_event, wpth)
    return cc_event


def load_kepler_curve(path, sparsen=True, verbose=False, plot=False):
    fits_image = fits.open(path)
    # The light curve is the second HDU in this FITS
    lc_hdu = fits_image[1]
    if verbose:
        print(fits_image.info())
        print(lc_hdu.data.names)



    # Read in the columns of data. For now, just focus on the SAP fluxes
    times_r = lc_hdu.data['time']
    sap_fluxes = lc_hdu.data['SAP_FLUX']
    sap_errors = lc_hdu.data['SAP_FLUX_ERR']

    # Remove all entries from ALL arrays where a NaN value is mapped

    sap_nans = np.argwhere(np.isnan(sap_fluxes))
    times = np.delete(times_r, sap_nans)[:1000]
    sap = np.delete(sap_fluxes, sap_nans)[:1000]
    errors = np.delete(sap_errors, sap_nans)[:1000]
    assert not np.isnan(sap).any()

    times_final = times
    sap_final = sap
    errors_final = errors
    if sparsen:
        # Remove random intervals
        #n = np.random.randint(2, 5)
        n = 1
        start_idx = np.random.randint(50, 500)
        times_final, sap_final, errors_final = times, sap, errors
        for i in range(n):
            #r = np.random.randint(10, 100)
            r = 400
            remove_idxs = np.arange(start_idx, start_idx+r)
            times_final = np.delete(times_final, remove_idxs)
            sap_final = np.delete(sap_final, remove_idxs)
            errors_final = np.delete(errors_final, remove_idxs)
            start_idx = np.random.randint(start_idx+100, start_idx+r)
        if verbose:
            print("Original length:", len(times))
            print("Sparse length:", len(times_final))

    if plot:
        # Plot
        plt.figure(figsize=(9, 4))
        plt.plot(times, sap, '-k', label='Original SAP Flux')
        if sparsen:
            plt.plot(times_final, sap_final, '-b', label='Sparse SAP Flux')
        plt.title('Kepler Light Curve')
        plt.legend()
        plt.xlabel('Time (days)')
        plt.ylabel('Flux (electrons/second)')
        plt.show()
    return times_final, sap_final, errors_final


def get_kepler_data(event_path, branch="Q17-small", sparsen=False):
    full_path = os.path.join(KEPLER_PATH, branch)
    print(full_path)
    kepler_data = []
    all_fluxes = []
    for root, dirs, files in tqdm(os.walk(full_path)):
        for file in files:
            curve_path = os.path.join(root, file)
            times, flux, errors = load_kepler_curve(curve_path, sparsen=sparsen)

            # Mean correct
            flux_mean = flux.mean(0)
            flux -= flux_mean

            all_fluxes.append(flux)
            save_path = None
            if event_path is not None:
                save_path = os.path.join(event_path, "{0}.pk".format(file))
            cc_event = make_cc_event(file, (times, flux, errors), save_path)
            kepler_data.append(cc_event)

    # Normalize
    #flux_std = np.array(all_fluxes).std()

    print("Loaded all light curves from", full_path)
    print("Got {0} light curves in total.".format(len(kepler_data)))
    return kepler_data


def get_cauchy_data(event_path, plot=False):
    labels = ["class1", "class2"]
    cauchy_data = []
    idx = 0
    for i in range(len(labels)):
        label_str = labels[i]
        with open(os.path.join(CAUCHY_PATH, "{0}.csv".format(label_str))) as cauchy_file:
            cauchy_reader = csv.reader(cauchy_file, delimiter="\t")
            for row in cauchy_reader:
                cauchy_np = np.array(list(map(lambda i: float(i), row)))
                timestamps = np.arange(cauchy_np.shape[0])
                save_path = None
                if event_path is not None:
                    save_path = os.path.join(event_path, "{0}.pk".format(idx))
                cc_cauchy = make_cc_event(idx, (timestamps, cauchy_np, None), save_path, label=str(i))
                cauchy_data.append(cc_cauchy)
                idx += 1
                if plot:
                    plt.figure(figsize=(9,4))
                    plt.plot(cauchy_np)
                    plt.xlabel("Time Steps")
                    plt.ylabel("Value")
                    plt.title("Cauchy curve of {0}".format(label_str))
    return cauchy_data


def get_coffee_data(event_path, plot=False):
    coffee_data = []
    idx = 0
    with open(os.path.join(COFFEE_PATH, "coffee.txt")) as coffee_file:
        for row in coffee_file.readlines():
            row = row.split()
            coffee_np = np.array(list(map(lambda i: float(i), row)))
            save_path = None
            if event_path is not None:
                save_path = os.path.join(event_path, "{0}.pk".format(idx))
            label = int(coffee_np[0])
            data = coffee_np[1:]
            timestamps = np.arange(data.shape[0])
            cc_cauchy = make_cc_event(idx, (timestamps, data, None), save_path, label=str(label))
            coffee_data.append(cc_cauchy)
            idx += 1
            if plot:
                plt.figure(figsize=(9, 4))
                plt.plot(data)
                plt.xlabel("Time Steps")
                plt.ylabel("Value")
                plt.title("Coffee curve of {0}".format(label))
    return coffee_data


def load_cc_events(events_path):
    cc_events = []
    for file in os.scandir(events_path):
        if file.is_file():
            with open(file, "rb") as cc:
                cc_event = pickle.load(cc)
                cc_events.append(cc_event)
    return cc_events


def load_data(dataset_name, event_path, sparsen=True):
    data = None
    if event_path is None or not os.path.exists(event_path):
        if event_path is not None:
            os.mkdir(event_path)
        if dataset_name.startswith("kepler"):
            data = get_kepler_data(event_path, dataset_name, sparsen=sparsen)
        elif dataset_name == "cauchy":
            data = get_cauchy_data(event_path)
        elif dataset_name == "coffee":
            data = get_coffee_data(event_path)
        else:
            raise ValueError("No such dataset:", dataset_name)
    else:
        data = load_cc_events(event_path)
    return data


# A unit test
if __name__ == "__main__":
    load_kepler_curve(os.path.join(KEPLER_PATH, "0007/000757076/kplr000757076-2013131215648_llc.fits"),
                      sparsen=True, verbose=True, plot=True)
    #get_kepler_data()

