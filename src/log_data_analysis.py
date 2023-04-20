import numpy as np

def linear_least_squares(x: np.ndarray, y: np.ndarray):
    """
    This function evaluates linear regression using the least squares method
    :param x: data
    :param y: data
    :return: coefficients and covariance matrix
    """
    # obtain coefficients
    n = len(x)
    if n < 3:
        raise ValueError('number of points should be at least three')
    X = np.hstack([np.ones(n)[:, None], x[:, None]])  # matrix for linear regression
    coefficients = np.linalg.pinv(X) @ y  # solve linear regression
    # obtain covariance
    P = X @ np.linalg.pinv(X)  # projection matrix
    M = np.eye(n) - P  # annihilator matrix
    p = 2  # dimension of parameter vector, 2 for linear regression
    sigma_squared = y.T @ M @ y / (n - p)  # estimate of the regression standard error
    # covariance = sigma_squared * np.linalg.inv(X.T @ X)  # covariance matrix
    return coefficients, sigma_squared


def log_estimate(x, y, w_min=0.01):
    """
    This function estimates parameters of exponent with the lowest an absolute degree and the greatest one
    :param x:
    :param y:
    :param w_min:
    :return:
    """
    cut = round(x.size / 4)
    cut2 = round(x.size / 6)

    coeffs, s = linear_least_squares(x[-cut:], np.log(y)[-cut:])
    D_1, w1 = -coeffs[1], np.exp(coeffs[0])
    if w1 >= 1:
        w1 = 0.9

    coeffs, s2 = linear_least_squares(x[:cut2], y[:cut2])
    D_n = -coeffs[1]
    return max(w1, w_min), max(D_1, 1e-4), min(D_n / w_min, 10), min(np.sqrt(s2), np.sqrt(s))


def bounds(D1, w1, D_max, n, w_min=0.01):
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
