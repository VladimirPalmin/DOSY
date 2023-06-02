from typing import Tuple

import numpy as np


def linear_least_squares(x: np.ndarray,
                         y: np.ndarray) -> np.ndarray:
    """
    This function evaluates linear regression using the least squares method.

    Parameters
    ----------
    x: np.ndarray
        X-values for the experiment. In original analysis it is
        normalized Z values.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.

    Returns
    -------
    coefficients: np.ndarray
        Coefficients for the linear model approximating the data.
        The inner structure is [intersection, slope].
    """
    # obtain coefficients
    n = len(x)
    if n < 3:
        raise ValueError('number of points should be at least three')
    # matrix for linear regression
    X = np.hstack([np.ones(n)[:, None], x[:, None]])
    # solve linear regression
    coefficients = np.linalg.pinv(X) @ y
    return coefficients


def log_estimate(x: np.ndarray,
                 y: np.ndarray,
                 w_min: float = 0.01) -> Tuple[float, float, float]:
    """
    Estimation of model parameters from the linear regression.

    Parameters
    ----------
    x: np.ndarray
        X-values for the experiment. In original analysis it is
        normalized Z values.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    w_min: float = 0.01
        Minimum value in the weight parameters of the model expected
        in the data.

    Returns
    -------
    w_est: float
        Estimation of exponent weight for single component model with the
        lowest self diffusion coefficient.
    D_est: float
        Estimation of exponent self-diffusion coefficient for single component
        model with the lowest self diffusion coefficient.
    D_max: float
        Estimation of maximum self-diffusion coefficient.
    """
    cut = round(x.size / 4)
    cut2 = round(x.size / 6)

    coeffs = linear_least_squares(x[-cut:], np.log(y)[-cut:])
    D_1, w1 = -coeffs[1], np.exp(coeffs[0])
    if w1 >= 1:
        w1 = 0.9

    coeffs = linear_least_squares(x[:cut2], y[:cut2])
    D_n = -coeffs[1]
    w_est, D_est, D_max = max(w1, w_min), max(D_1, 1e-4), min(D_n / w_min, 10)
    return w_est, D_est, D_max


def bounds(D1: float,
           w1: float,
           D_max: float,
           n: int,
           w_min: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimation of function bounds and initial guess for given number of
    components.

    Parameters
    ----------
    D1: float
        Estimation of exponent self-diffusion coefficient for single component
        model with the lowest self diffusion coefficient.
    w1: float
        Estimation of exponent weight for single component model with the
        lowest self diffusion coefficient.
    D_max: float
        Estimation of maximum self-diffusion coefficient.
    n: int
        Number of components to generate the bounds and initial guess for.
    w_min: float = 0.01
        Minimum value in the weight parameters of the model expected
        in the data.

    Returns
    -------
    x0: np.ndarray
        Initial guess for the parameters for given number of components.
        Even elements (0, 2, ...) represent weights of exponents.
        Odd elements (1, 3, ...) - self-diffusion coefficients.
    xl: np.ndarray
        Lower bound for the parameters for given number of components.
        Even elements (0, 2, ...) for weights of exponents.
        Odd elements (1, 3, ...) for self-diffusion coefficients.
    xw: np.ndarray
        Upper bound for the parameters for given number of components.
        Even elements (0, 2, ...) for weights of exponents.
        Odd elements (1, 3, ...) for self-diffusion coefficients.
    """
    D_min = D1 * 0.5
    w_max = 1 - w1
    # initial guess
    Ds = np.linspace(D1, D_max * 0.9, n // 2)
    ws = np.zeros(n // 2)
    ws[0] = w1
    if w_max * 0.8 <= w_min * 1.1:
        ws[1:] = w_min * 1.1
    else:
        ws[1:] = np.linspace(w_min * 1.1, w_max * 0.9, n // 2 - 1)
    x0 = np.zeros(n)
    x0[::2] = ws
    x0[1::2] = Ds
    # lower bound
    xl = np.zeros(n)
    xl[::2] = w_min
    xl[1::2] = D_min
    # upper bound
    xw = np.zeros(n)
    xw[::2] = 1
    xw[1::2] = D_max
    return x0, xl, xw
