"""Feature extraction.

This file provides a number of functions for feature extraction from given
data window, agnostic of size and method of delivery.
"""

import numpy as np
import math
import scipy.stats as st


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

# TODO 2nd-order butterworth high-pass

# TODO other features from activity detection project

# TODO feature extract function to pull everything together
