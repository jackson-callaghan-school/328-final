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
    return np.std(mag_win(window), axis=0)


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

def freq_peak_win(window):
    """Return a list of frequencies which appear as peaks in fft of the window

    Returns an empty list if there are no peaks.
    """
    mag = mag_win(window)

    sig = np.fft.rfft(mag, axis=0)
    rsig = sig.real.astype(float)
    ind, _ = find_peaks(rsig, prominence=1)

    if len(ind) == 0:
        return []
    else:
        return [rsig[i] for i in ind]

# TODO 2nd order butterworth highpass

# TODO feature extract function to pull everything together
