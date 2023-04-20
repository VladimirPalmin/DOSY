import numpy as np
from scipy import optimize

from src.log_data_analysis import log_estimate, bounds

def sigmoid(x, border=1.6):
    return 1 / (1 + np.exp(-3 * (x-border)))


def error_estimate(x, sigma=0.018):
    return ((0.018*np.exp(-0.94 * x)+0.0002) * sigmoid(x) + (0.013*np.exp(-6 * x)+0.0051) * (1 - sigmoid(x)))/0.018 * sigma


def sum_exp(params, x):
    """
    Exponent sum
    :param params: Starting with 1, odd params - weights of exponents, even params - coefficients
    """
    res = np.zeros(len(x))
    for i in range(0, len(params), 2):
        res += params[i] * np.exp(-x * params[i + 1])
    return res


def sum_exp_curv(x, *params):
    return sum_exp(params, x)


def least_sqruares(y, y_pred):
    return np.linalg.norm(y_pred - y)


def chi_square(y, y_pred, sigma):
    dof = len(y)
    return np.sum((y - y_pred) ** 2 / sigma ** 2) / dof


def gen_data(seed, params, n=225, sigma=0.001):
    np.random.seed(seed)
    x = np.geomspace(0.1, 6, n)
    y = np.random.normal(loc=sum_exp(params, x), scale=sigma)
    y[y < 0] = -y[y < 0]
    return x, y


def right_order(params):
    """
    This function makes the order of exponents in the right way:
    little coefficients are first
    """
    sort_params = np.zeros(len(params))
    sort_ind = np.argsort(params[1::2])
    sort_params[1::2] = params[2 * sort_ind + 1]
    sort_params[::2] = params[2 * sort_ind]
    return sort_params


def loss(params, x, y):
    return sum_exp(params, x) - y


def loss_function(params, x, y, reg=0, sigma=None, func=sum_exp):
    if sigma is None:
        sigma = 1
    y_pred = func(params, x)
    return chi_square(y, y_pred, sigma) + reg * np.linalg.norm(params)


def fit(x, y, n, x0=None, method='BFGS', reg=0.0, sigma=None):
    w1, D1, D_max, s = log_estimate(x, y)
    _x0, xl, xw = bounds(D1, w1, D_max, 2 * n)
    if type(sigma) is float:
        sigma = error_estimate(x, sigma)

    if x0 is None:
        x0 = _x0

    if method == 'least_squares':
        params = optimize.least_squares(loss, x0=x0, args=(x, y),
                                        bounds=list(zip(xl, xw)),
                                        method=method,
                                        tr_solver='exact',
                                        tr_options={'regularize': True}).x
        params = right_order(params)

    if method == 'curve_fit':
        params, pcov = optimize.curve_fit(sum_exp_curv, x, y, p0=x0, bounds=(xl, xw),
                                          maxfev=100000)
        params = right_order(params)

    elif method == 'dual_annealing':
        res = optimize.dual_annealing(loss_function, bounds=list(zip(xl, xw)), x0=x0,
                                      args=(x, y, reg, sigma),
                                      seed=42, initial_temp=1, maxiter=1000, visit=2, accept=-1,
                                      no_local_search=False,
                                      minimizer_kwargs={'method': 'BFGS'})
        params = right_order(res.x)

    elif method == 'BFGS':
        res = optimize.minimize(loss_function, x0=x0,
                                args=(x, y, reg, sigma), method='BFGS')
        params = right_order(res.x)

    elif method == 'L-BFGS-B':
        res = optimize.minimize(loss_function, x0=x0, bounds=list(zip(xl, xw)),
                                args=(x, y, reg, sigma), method='L-BFGS-B')
        params = right_order(res.x)

    else:
        raise ValueError('method should be curve_fit, dual_annealing, L-BFGS-B, or BFGS')
    return params


def fits(x, y, n_min=1, n_max=5, method='curve_fit', reg=0.0, boost=False, sigma=None):
    params_est = []
    x0 = None
    if boost:
        w1, D1, D_max, s = log_estimate(x, y)
        x0, _, _ = bounds(D1, w1, D_max, 2 * n_min)
    for n in range(n_min, n_max + 1):
        params = fit(x, y, n, x0, method, reg, sigma)
        params_est.append(right_order(params))
        if boost:
            x0 = np.zeros(2 * (n + 1))
            x0[:-2] = right_order(params)
    return params_est
