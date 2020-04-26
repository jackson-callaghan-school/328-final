"""Feature extraction.

This file provides a number of functions for feature extraction from given
data window, agnostic of size and method of delivery.
"""

import numpy as np
import math
import scipy.stats as st
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter
from scipy.signal import find_peaks


def mag_win(window):
    """Compute the magnitude signal over a window.

    Args:
        window (array_like): an array of x,y,z.

    Returns:
        array: a 1d array of magnitude signal.

    """
    return [np.sqrt(x**2+y**2+z**2) for x, y, z in window]


def mean_win(window):
    """
    Compute the mean x, y and z over the given window.
    """
    return np.mean(window, axis=0)


def median_win(window):
    """
    Compute median x, y, and z over given window.
    """
    return np.median(window, axis=0)


def stdev_win(window):
    """Compute the standard deviation of x, y, and z over window.
    """
    return np.std(window, axis=0)


def entropy_win(window):
    """Compute the entropy of a given window.
    """

    mag = mag_win(window)

    hist, bin_edges = np.histogram(mag, bins=5, density=True)

    # calculate entropy here

    prob = np.diff(bin_edges)*hist
    entropy = 0
    for p in prob:
        if (p > 0):
            entropy += p*math.log2(p)

    return -entropy


def max_mag_win(window):
    """Compute the maximum magnitude in the window
    """
    mag = mag_win(window)
    return np.amax(mag)


def npeaks_win(window):
    """Compute the number of peaks in the window
    """
    mag = mag_win(window)
    ind, _ = find_peaks(mag, height=np.mean(mag)+1, prominence=1)
    return len(ind)


def fft_npeak_win(window):
    """Return a list of frequencies which appear as peaks in fft of the window

    Returns an empty list if there are no peaks.
    """
    mag = mag_win(window)

    sig = np.fft.rfft(mag, axis=0)
    rsig = sig.real.astype(float)
    ind, _ = find_peaks(rsig, prominence=1)

    return len(ind)


def butter2h_win(window):
    """Return the window passed through a 2nd-order butterworth high-pass.
    """
    # parameters
    order = 2
    fs = 100000  # maybe wrong
    cutoff = 5  # TODO needs to be dialed in
    # filter
    nyq = 0.5 * fs
    normal_cuttoff = cutoff / nyq
    b, a = butter(order, normal_cuttoff, btype='high', analog=False)

    return (filtfilt(b, a, window[0]),
            filtfilt(b, a, window[1]),
            filtfilt(b, a, window[2]))  # possible done incorrectly

# TODO feature extract function to pull everything together


def extract_features(window):

    window_a = window[0:3]
    window_g = window[3:6]

    x = []
    feature_names = []

    # accelerometer

    x.append(mean_win(window_a))
    feature_names.append("x_mean_a_g")
    feature_names.append("y_mean_a")
    feature_names.append("z_mean_a")

    x.append(median_win(window_a))
    feature_names.append("x_med_a")
    feature_names.append("y_med_a")
    feature_names.append("z_med_a")

    x.append(stdev_win(window_a))
    feature_names.append("x_stdev_a")
    feature_names.append("y_stdev_a")
    feature_names.append("z_stdev_a")

    x.append(mean_win(mag_win(window_a)))
    x.append(median_win(mag_win(window_a)))
    x.append(stdev_win(mag_win(window_a)))
    feature_names.append("mag_mean_a")
    feature_names.append("mag_med_a")
    feature_names.append("mag_stdev_a")

    x.append(max_mag_win(window_a))
    feature_names.append("mag_max_a")

    x.append(entropy_win(window_a))
    feature_names.append("entropy_a")

    x.append(npeaks_win(window_a))
    feature_names.append("npeaks_a")

    x.append(fft_npeak_win(window_a))
    feature_names.append("fft_npeaks_a")

    # gyroscope features

    x.append(mean_win(window_g))
    feature_names.append("x_mean_g")
    feature_names.append("y_mean_g")
    feature_names.append("z_mean_g")

    x.append(median_win(window_g))
    feature_names.append("x_med_g")
    feature_names.append("y_med_g")
    feature_names.append("z_med_g")

    x.append(stdev_win(window_g))
    feature_names.append("x_stdev_g")
    feature_names.append("y_stdev_g")
    feature_names.append("z_stdev_g")

    x.append(mean_win(mag_win(window_g)))
    x.append(median_win(mag_win(window_g)))
    x.append(stdev_win(mag_win(window_g)))
    feature_names.append("mag_mean_g")
    feature_names.append("mag_med_g")
    feature_names.append("mag_stdev_g")

    x.append(max_mag_win(window_g))
    feature_names.append("mag_max_g")

    x.append(entropy_win(window_g))
    feature_names.append("entropy_g")

    x.append(npeaks_win(window_g))
    feature_names.append("npeaks_g")

    x.append(fft_npeak_win(window_g))
    feature_names.append("fft_npeaks_g")

    # add any other features here
    # I didn't do anything with the butterworth filter
    # might be worth running its output into other features

    feature_vector = np.concatenate(x, axis=0)

    return feature_names, feature_vector
