from itertools import combinations
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import optimize

from src.log_data_analysis import bounds, log_estimate


def sigmoid(x: np.ndarray,
            border: float = 1.6,
            spread: float = 3.0) -> np.ndarray:
    """
    Sigmoid function with adjustable parameters.

    Parameters
    ----------
    x: np.ndarray
        Array of points to calculate sigmoid in.
    border: float = 1.6
        Value of the middle of sigmoid.
    spread: float = 3.0
        Coefficient of steepness of the sigmoid.

    Returns
    -------
    np.ndarray
        Values of sigmoid in given points.
    """

    return 1 / (1 + np.exp(- spread * (x-border)))


def error_estimate(x: np.ndarray,
                   sigma: float = 0.018,
                   est_coeff: float = 1.1) -> np.ndarray:
    """
    Calculation of error profile described in the original work.

    Parameters
    ----------
    x: np.ndarray
        Array of points to calculate error profile in.
    sigma: float = 0.018
        Maximum absolute value of the error profile.
    est_coeff: float = 1.1
        Estimation of the self-diffusion coefficient in the model
        for the horizontal scaling.

    Returns
    -------
    np.ndarray
        Error profile in the given points with the given scaling.
    """
    x1 = x * est_coeff / 1.1 
    return ((0.018 * np.exp(-0.94 * x1) + 0.0002) * sigmoid(x1) +
            (0.013 * np.exp(-6 * x1) + 0.0051) * (1 - sigmoid(x1))
            )/0.018 * sigma


def sum_exp(params: np.ndarray,
            x: np.ndarray) -> np.ndarray:
    """
    Calculation of the model normalized I values with given parameters
    in given points.

    Parameters
    ----------
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.

    Returns
    -------
    res: np.ndarray
        Array of the model normalized I values in given points.
    """
    res = np.zeros(len(x))
    for i in range(0, len(params), 2):
        res += params[i] * np.exp(-x * params[i + 1])
    return res


def _sum_exp_curv(x: np.ndarray,
                  *params: float) -> np.ndarray:
    """
    Alias for sum_exp function with different argument passing.
    Used in the curve_fit function.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    *params: float
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.

    Returns
    -------
    np.ndarray
        Array of the model normalized I values in given points.
    """
    return sum_exp(params, x)


def chi_square(y: np.ndarray,
               y_pred: np.ndarray,
               sigma: Union[np.ndarray, float]) -> float:
    """
    Calculation of reduced chi-square value for given points.

    Parameters
    ----------
    y: np.ndarray
        Experimental points.
    y_pred: np.ndarray
        Model points.
    sigma: np.ndarray or float
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
    float
        Reduced chi-square value for given parameters.
    """
    dof = len(y)
    return np.sum((y - y_pred) ** 2 / sigma ** 2) / dof


def right_order(params: np.ndarray) -> np.ndarray:
    """
    Reordering the parameters to put the components in ascending order
    by self-diffusion coefficient.

    Parameters
    ----------
    params: np.ndarray
        1-d array of floats with parameters of the model.

    Returns
    -------
    sort_params: np.ndarray
        1-d array of floats with reorganized parameters of the model.
    """
    sort_params = np.zeros(len(params))
    sort_ind = np.argsort(params[1::2])
    sort_params[1::2] = params[2 * sort_ind + 1]
    sort_params[::2] = params[2 * sort_ind]
    return sort_params


def loss(params: np.ndarray,
         x: np.ndarray,
         y: np.ndarray) -> np.ndarray:
    """
    Calculation of residuals of model and data.

    Parameters
    ----------
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.

    Returns
    -------
    np.ndarray
        Array of residuals between the experimental data and model.
    """
    return sum_exp(params, x) - y


def loss_function(params: np.ndarray,
                  x: np.ndarray,
                  y: np.ndarray,
                  reg: float = 0.0,
                  sigma: Optional[Union[float, np.ndarray]] = None,
                  func: Callable = sum_exp) -> float:
    """
    Loss function with chi-square and L2-regularization.

    Parameters
    ----------
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    reg: float = 0.0
        Weight-coefficient for the regularizaion component of loss function.
    sigma: float or np.ndarray or None
        Expected deviation in experimental points.
        If float same sigma is used for every point.
        If None chi-square is reduced to simple least squares expression.
    func: callable = sum_exp
        Function to model the Y data from X data.

    Returns
    -------
    float
        Value of loss-function with given parameters.
    """
    if sigma is None:
        sigma = 1
    y_pred = func(params, x)
    return chi_square(y, y_pred, sigma) + reg * np.linalg.norm(params)


def fit(x: np.ndarray,
        y: np.ndarray,
        n: int,
        x0: Optional[np.ndarray] = None,
        method: str = 'BFGS',
        reg: float = 0.0,
        sigma: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """
    Estimation of the model parameters with Weighted least squares (WLS)
    estimator for a given number of components.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    n: int
        Number of components to use for the model.
    x0: np.ndarray or None
        Initial guess for the parameters of the model.
        If None the initial guess will be generated with the bounds function.
    method: str = 'BFGS'
        Method to choose for the optimization procedure. Must be on of the list
        {least_sqaures, curve_fit, dual_annealing, L-BFGS-B, BFGS}.
    reg: float = 0.0
        Weight-coefficient for the regularizaion component of loss function.
        Ignored in 'least_squares' and 'curve_fit'.
    sigma: float or np.ndarray or None
        Expected deviation in experimental points.
        If float the array is generated with error_estimate with given
        maximum value.
        If None chi-square is reduced to simple least squares expression.
        Ignored in 'least_squares' and 'curve_fit'.

    Returns
    -------
    params: np.ndarray
        Parameters of the model calculated with the estimator for the given
        number of components.
        Even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    -------
    """
    w1, D1, D_max = log_estimate(x, y)
    _x0, xl, xw = bounds(D1, w1, D_max, 2 * n)
    if type(sigma) is float:
        est_coeff = -(np.log(y[0]) - np.log(y[-1]))/(x[0] - x[-1])
        sigma = error_estimate(x, sigma, est_coeff)

    if x0 is None:
        x0 = _x0

    if method == 'least_squares':
        params = optimize.least_squares(loss, x0=x0, args=(x, y),
                                        bounds=list(zip(xl, xw)),
                                        tr_solver='exact',
                                        tr_options={'regularize': True}).x
        params = right_order(params)

    if method == 'curve_fit':
        params, pcov = optimize.curve_fit(_sum_exp_curv, x, y, p0=x0, bounds=(xl, xw),
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
        raise ValueError('method should be least_sqaures, curve_fit, '
                         'dual_annealing, L-BFGS-B, or BFGS')
    return params


def fits(x: np.ndarray,
         y: np.ndarray,
         n_min: int = 1,
         n_max: int = 5,
         method: str = 'BFGS',
         reg: float = 0.0,
         boost: bool = False,
         sigma: Optional[Union[float, np.ndarray]] = None) -> list:
    """
    Estimation of the model parameters with Weighted least squares (WLS)
    estimator for several models with different numbers of combinations.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    n_min: int = 1
        Minimum number of components to calculate parameters for.
    n_max: int = 5
        Maximum number of components to calculate parameters for.
    method: str = 'BFGS'
        Method to choose for the optimization procedure. Must be on of the list
        {least_sqaures, curve_fit, dual_annealing, L-BFGS-B, BFGS}.
    reg: float = 0.0
        Weight-coefficient for the regularizaion component of loss function.
        Ignored in 'least_squares' and 'curve_fit'.
    boost: bool = False
        If True the parameters estimated for model with n components are
        used as initial guess for estimator for model witn (n+1) components.
    sigma: float or np.ndarray or None
        Expected deviation in experimental points.
        If float the array is generated with error_estimate with given
        maximum value.
        If None chi-square is reduced to simple least squares expression.
        Ignored in 'least_squares' and 'curve_fit'.

    Returns
    -------
    params_est: list
        List with the np.ndarray elements. i-th element shows the estimated
        parameters for the model with (i+1) components. In each element of list
        even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    """
    params_est = []
    x0 = None
    if boost:
        w1, D1, D_max = log_estimate(x, y)
        x0, _, _ = bounds(D1, w1, D_max, 2 * n_min)
    for n in range(n_min, n_max + 1):
        params = fit(x, y, n, x0, method, reg, sigma)
        params_est.append(right_order(params))
        if boost:
            x0 = np.zeros(2 * (n + 1))
            x0[:-2] = right_order(params)
    return params_est


def loss_function_baes(params: np.ndarray,
                       x: np.ndarray,
                       y: np.ndarray,
                       params_baes: np.ndarray,
                       params_baes_sigma: np.ndarray,
                       sigma: Optional[Union[float, np.ndarray]] = None,
                       func: Callable = sum_exp) -> float:
    """
    Loss function for Maximum aposteriori estimator.

    Parameters
    ----------
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    params_baes: np.ndarray
        Array with bayesian estimation of parameters of the model.
        Must be the same size as the params argument.
    params_baes_sigma: np.ndarray
        Array with bayesian estimation of deviation in parameters of the model.
        Must be the same size as the params argument.
    sigma: float or np.ndarray or None
        Expected deviation in experimental points.
        If float same sigma is used for every point.
        If None chi-square is reduced to simple least squares expression.
    func: callable = sum_exp
        Function to model the Y data from X data.

    Returns
    -------
    float
        Value of loss-function with given parameters.
    """
    y_pred = func(params, x)
    return (chi_square(y, y_pred, sigma) +
            chi_square(params[1::2], params_baes[1::2], params_baes_sigma[1::2]) +
            chi_square(np.array([params[::2].sum()]), np.ones(1) * 1, 0.01))


def fit_baes(x: np.ndarray,
             y: np.ndarray,
             n: int,
             params_baes: np.ndarray,
             params_baes_sigma: np.ndarray,
             x0: Optional[np.ndarray] = None,
             sigma: Optional[Union[float, np.ndarray]] = None,
             param_bounds: Optional[list] = [[-0.1, 1.1], [0, 3]]
             ) -> Tuple[np.ndarray, float]:
    """
    Estimation of the model parameters with Maximum a posteriori (MAP)
    estimator for a given number of components.

    Parameters
    ----------
    x: np.ndarray,
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray,
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    n: int,
        Number of components to use for the model.
    params_baes: np.ndarray,
        Array with bayesian estimation of parameters of the model.
        Must be the same size as the params argument.      
    params_baes_sigma: np.ndarray,
        Array with bayesian estimation of deviation in parameters of the model.
        Must be the same size as the params argument.
    x0: (np.ndarray or None) = None
        Initial guess for the parameters of the model.
        If None the initial guess will be generated with the bounds function.
    sigma: (float or np.ndarray or None) = None
        Expected deviation in experimental points.
        If float the array is generated with error_estimate with given
        maximum value.
        If None chi-square is reduced to simple least squares expression.
    param_bounds: (list or None) = [[-0.1, 1.1], [0, 3]]
        List of 2 lists. Bounds for the parameter space used in
        L-BFGS-B optimization. The structure for the argument is
        [[w_min, w_max], [D_min, D_max]], where w is weight of the exponent,
        D is a self-diffusion coefficient of the exponent.

    Returns
    -------
    params: np.ndarray
        Parameters of the model calculated with the estimator for the given
        number of components.
        Even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    float
        The value of loss function after minimization.
    """
    w1, D1, D_max = log_estimate(x, y)
    _x0, xl, xw = bounds(D1, w1, D_max, 2 * n)
    if type(sigma) is float:
        est_coeff = -(np.log(y[0]) - np.log(y[-1]))/(x[0] - x[-1])
        sigma = error_estimate(x, sigma, est_coeff)
    if x0 is None:
        x0 = _x0
    if param_bounds is not None:
        res = optimize.minimize(loss_function_baes, x0=x0,
                                args=(x, y,
                                      params_baes, params_baes_sigma,
                                      sigma),
                                method='L-BFGS-B',
                                bounds=param_bounds * n)
    else:
        res = optimize.minimize(loss_function_baes, x0=x0,
                                args=(x, y,
                                      params_baes, params_baes_sigma,
                                      sigma),
                                method='BFGS')
    params = right_order(res.x)
    return params, res.fun


def fits_baes(x: np.ndarray,
              y: np.ndarray,
              sigma: Union[float, np.ndarray],
              params_baes: np.ndarray,
              params_baes_sigma: Union[np.ndarray, float],
              mode: str = 'base',
              n_min: int = 1,
              n_max: int = 4,
              boost: bool = False,
              param_bounds: Optional[list] = [[-0.1, 1.1], [0, 3]]):
    """
    Estimation of the model parameters with Maximum a posteriori (MAP)
    estimator for several models with different numbers of combinations.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    sigma: float or np.ndarray
        Expected deviation in experimental points.
        If float the array is generated with error_estimate with given
        maximum value.
    params_baes: np.ndarray
        if mode == 'experiment'
        Array of arrays with bayesian estimation of parameters of the model.
        i-th array of the argument must represent the parameters for estimation
        of the model with (i+1) components.
        if mode == 'base'
        Array with the self-diffusion coefficients expected in the data.      
    params_baes_sigma: np.ndarray or float,
        If mode == 'experiment'
        Array of arrays with bayesian estimation deviation for parameters
        of the model. i-th array of the argument must represent the deviation
        in parameters for estimation of the model with (i+1) components.
        If type is float the value is used as deviation for all self-diffusion
        coefficients.
        If mode == 'base'
        Only float can be passed. The value is used as deviation for all
        self-diffusion coefficients.
    n_min: int = 1
        Minimum number of components to calculate parameters for.
    n_max: int = 4
        Maximum number of components to calculate parameters for.
    boost: bool = False
        If True the parameters estimated for model with n components are
        used as initial guess for estimator for model witn (n+1) components.
    param_bounds: (list or None) = [[-0.1, 1.1], [0, 3]]
        List of 2 lists. Bounds for the parameter space used in
        L-BFGS-B optimization. The structure for the argument is
        [[w_min, w_max], [D_min, D_max]], where w is weight of the exponent,
        D is a self-diffusion coefficient of the exponent.

    Results
    -------
    params_est: list
        List with the np.ndarray elements. i-th element shows the estimated
        parameters for the model with (i+1) components. In each element of list
        even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    """
    params_est = []
    x0 = None
    if boost:
        w1, D1, D_max = log_estimate(x, y)
        x0, _, _ = bounds(D1, w1, D_max, 2 * n_min)

    if mode == 'experiment':
        sigma_scaling = 5
        if type(params_baes_sigma) is float:
            params_baes_sigma = np.array([np.array(i*[0, params_baes_sigma])
                                          for i in range(n_min, n_max+1)])
        for n in range(n_min, n_max + 1):
            params, _ = fit_baes(x, y, n, params_baes[n-1],
                                 sigma_scaling * params_baes_sigma[n-1],
                                 x0,
                                 sigma,
                                 param_bounds)
            params_est.append(right_order(params))
            if boost:
                x0 = np.zeros(2 * (n + 1))
                x0[:-2] = right_order(params)
        return params_est

    elif mode == 'base':
        for n in range(n_min, n_max + 1):
            params = -1 * np.ones(2*n)
            fun_val = np.inf
            for self_diff in combinations(params_baes, n):
                curr_baes, curr_baes_sigma = np.zeros((2, 2 * n))
                curr_baes[::2] = 1/n
                curr_baes[1::2] = sorted(self_diff)
                curr_baes_sigma[1::2] = params_baes_sigma
                temp_params, temp_fun_val = fit_baes(x, y, n,
                                                     curr_baes,
                                                     curr_baes_sigma,
                                                     x0,
                                                     sigma=sigma,
                                                     param_bounds=param_bounds)
                if temp_fun_val < fun_val:
                    params, fun_val = temp_params, temp_fun_val
            params_est.append(right_order(params))
            if boost:
                x0 = np.zeros(2 * (n + 1))
                x0[:-2] = right_order(params)
        return params_est
    else:
        raise ValueError('Mode must be "experiment" or "base"!')