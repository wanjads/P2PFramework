import numpy as np

"""
Contains utilities for displaying the results of the simulation.
"""



def running_mean(episode_no, m, new_cost):
    """Calculates a running mean.

    Parameters
    ----------
    episode_no : int
        The index of the element to included in the mean. episode_no - 1 number of elements are already included.
    m : float
        The old mean value.
    new_cost : float
        The value of the element to include in the mean.

    Returns
    -------
     : float
        The new mean.

    """
    return (episode_no / (episode_no + 1)) * m + (1 / (episode_no + 1)) * new_cost


def moving_average(data, window):
    """Calculates a moving average of the data with a specific window size.

    Parameters
    ----------
    data : ndarray
        The data to calculate the moving average on. 1D
    window : int
        The window size used for calculation.

    Returns
    -------
    data_ma : ndarray
        The moving average of each index from data[window - 1] to the end.

    """
    if window > len(data):
        raise ValueError('The window of the running mean must be greater than the length of the data.')
    data_ma = np.zeros((len(data) - window + 1))
    for i in range(window - 1, len(data)):
        j = i - window + 1
        data_ma[j] = np.sum(data[i - window + 1:i + 1]) / window
    return data_ma
