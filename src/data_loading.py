import numpy as np

from typing import Tuple


def load_data(link_to_data: str,
              scale: float = 1e6,
              skiprows: int = 2,
              usecols: tuple = (1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loading I and Z data from the txt file and normalization.

    Parameters
    ----------
    link_to_data: str
        Link to txt file with the data.
    scale: float = 1e6
        Normalizing coefficient for Z values.
    skiprows: int = 2
        Number of rows in file to skip.
    usecols: tuple = (1, 2)
        Indexes of columns to use.

    Returns
    -------
    x: np.ndarray
        Z values for the experiment normalized to scale argument.
    y: np.ndarray
        I values for the experiment normalized to the first I value.
    """
    x, y = np.loadtxt(link_to_data, skiprows=skiprows, usecols=usecols).T
    # we consider I/I0
    y = y / y[0]
    x = x / scale
    return x, y
