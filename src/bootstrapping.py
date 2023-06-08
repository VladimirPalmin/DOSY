import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from src.mixture_fit import error_estimate
from src.optimal_number import optimal_params


def conf_intervals(params: np.ndarray,
                   sigmas: np.ndarray,
                   level: float = 2.0) -> np.ndarray:
    """
    Estimation of level-confidence intervals for 1d array.

    Parameters
    ----------
    params: np.ndarray
        1-d array with parameters of the model.
    sigmas: np.ndarray
        1-d array with estimated 1-sigma deviation for model parameters.
    level: float
        Coefficint to define the level of confidence. E.g. level=2.0
        represents the 2-sigma confidence.

    Returns
    -------
    intervals: np.ndarray
        Array with shape (len(params), 2).
        Each element of array represents lower and upper
        confidence intervals for parameter respectively.
    """
    if len(params) != len(sigmas):
        raise RuntimeError('params and sigmas must be the same length!')

    intervals = np.zeros((len(params), 2), dtype=object)
    for i in range(len(params)):
        intervals[i] = (params[i] - level * sigmas[i],
                        params[i] + level * sigmas[i])
    return intervals


def check_similarity(params: np.ndarray,
                     intervals: np.ndarray) -> bool:
    """
    Check if any self-diffusion coefficicents are statistically similar.

    Parameters
    ----------
    params: np.ndarray
        1-d array with parameters of the model.
    intervals: np.ndarray
        2-d array with shape (len(params), 2) representing confidence intervals
        for every parameter.

    Returns
    -------
    result: bool
        True if any self-diffusion coefficients are inside any other
        confidence interval.
        False otherwise.
    """
    for param in params[1::2]:
        entries = np.sum([1 if interval[0] < param < interval[1] else 0
                          for interval in intervals[1::2]])
        if entries > 1:
            return True
    return False


def final_guess(x: np.ndarray,
                y: np.ndarray,
                sigma: Union[np.ndarray, float],
                params: np.ndarray,
                params_std: np.ndarray,
                conf_level: float = 2.0) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Estimate the number of components based on statistical properties
    of parameters. The number is estimated with AIC, BIC, then checked
    for components with similar self-diffusion coefficients,
    components with negative parameters, and components with
    weights compatible with zero.

    Parameters
    ----------
    x: np.ndarray
        X-values for the experiment. In original analysis it is
        normalized Z values.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    sigma: np.ndarray or float
        1-d array with lenght = len(x). Represents the estimated errors
        in y.
        If float same sigma is used for every point.
    params: np.ndarray
        Array of arrays with the model parameters. i-th element of array
        represents the parameters for a (i+1) number of components.
    params_std: np.ndarray
        Array of arrays with the deviation of the model parameters.
        i-th element of array represents the parameters for a (i+1) number
        of components.
    conf_level: float
        Coefficient to define the level of confidence. E.g. level=2.0
        represents the 2-sigma confidence.

    Returns
    -------
    indx: int
        index of the optimal parameter combination in params.
        Equals to (optimal number of components - 1).
    params_opt: np.ndarray
        array with optimal params, equals params[indx]
    params_opt_std: np.ndarray
        array with deviations of optimal params, equals params_std[indx]
    """
    # aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_idx
    _, _, _, _, _, _, cons_idx = optimal_params(x, y, params, sigma)

    params_opt = params[cons_idx]
    params_opt_std = params_std[cons_idx]
    indx = cons_idx

    intervals = conf_intervals(params_opt, params_opt_std, conf_level)
    check_sim = check_similarity(params_opt, intervals)
    check_negative = np.any(params_opt < 0)
    check_zero = np.any(intervals.T[0][::2] < 0)
    check = check_sim or check_negative or check_zero
    while check and indx > 0:
        indx = indx - 1
        params_opt = params[indx]
        params_opt_std = params_std[indx]

        intervals = conf_intervals(params_opt, params_opt_std, conf_level)
        check_sim = check_similarity(params_opt, intervals)
        check_negative = np.any(params_opt < 0)
        check_zero = np.any(intervals.T[0][::2] < 0)
        check = check_sim or check_negative or check_zero
    return indx, params_opt, params_opt_std


def bootstrap(function: callable,
              x: np.ndarray,
              y_model: np.ndarray,
              calc_sigma: float,
              num: int = 100,
              conf_level: float = 2.0,
              show_progress: bool = False,
              *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs bootstrapping analysis with given estimator for given
    model I values and noise level. Estimates both parameters for all
    component combinations and optimal number of parameters (see final_guess)

    Parameters
    ----------
    function: callable
        The estimator to be used in the bootstrappig.
        Note that it is expected to iterate through the several models
        for proper analysis. For reference see the 'fits' function in
        mixture_fit.py.
    x: np.ndarray
        X-values for the experiment. In original analysis it is
        normalized Z values.
    y_model: np.ndarray
        Ideal model Y-values for the experiment. In original analysis it is
        simulated normalized I values. Represent the model values of Y
        without any noise.
    calc_sigma: float
        Estimation for the maximum noise level in the Y data.
        Typical value for the dataset used in the original analysis is 0.018.
    num: int = 100
        Number of generated experiments to perform in boostrapping.
    conf_level: = 2.0
        Coefficient to define the level of confidence in final guess.
        E.g. level=2.0 represents the 2-sigma confidence.
    show_progress: bool = False
        If True shows the progress bars for the bootstrapping process.
    *args, **kwargs
        Used to pass arguments to function. Avoid repetitions
        with bootstrapping function arguments.

    Returns
    -------
    results: np.ndarray
        Array of complete parameter estimation for all simulated datasets.
        Each element is an array of results of estimation with different
        number of components.
    optim_results: np.ndarray
        Array with the optimal estimation results for all simulated datasets.
        Each element consists of (optimal index, optimal parameters, std of
        optimal parameters).
        Optimal parameters are calculated with the final_guess function.
    """

    est_coeff = -(np.log(y_model[0]) - np.log(y_model[-1]))/(x[0] - x[-1])
    calc_sigma = error_estimate(x, calc_sigma, est_coeff)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        samples = list(map(
            lambda _: (x, y_model + np.random.normal(0, calc_sigma, len(x))),
            range(num))
            )

        futures = [pool.submit(function, x1, y1, *args, **kwargs)
                   for x1, y1 in samples]

        results = []
        for future in tqdm(as_completed(futures),
                           total=num,
                           disable=(not show_progress)):
            results.append(future.result())
        results = np.vstack(results)
        params_std = ((results - results.mean(0)) ** 2).mean(0) ** 0.5

        optim_futures = [pool.submit(final_guess, x1, y1, calc_sigma,
                                     params, params_std, conf_level=conf_level)
                         for (x1, y1), params in zip(samples, results)]
        optim_results = []
        for future in as_completed(optim_futures):
            optim_results.append(future.result())
        optim_results = np.vstack(optim_results)

        return results, optim_results


def plot_parameters_space(params: np.ndarray,
                          bins: int = 30,
                          cut: float = 0.0) -> None:
    """
    Function to plot the distributions of estimated parameters for models
    with the same number of components. Note that the array must be 2-d to
    be processed (try np.vstack for turning array of arrays to 2-d array).

    Parameters
    ----------
    params: np.ndarray
        2-d array with shape (num of results, num of parameters).
        The array with parameters for models with the same num of components.
    bins: int = 30
        Number of bins to divide data into.
    cut: float = 0.0
        Percentile of points to ignore in plotting procedure.
        Must be a float in range [0, 1].
    """
    fig, ax = plt.subplots(params.shape[1], params.shape[1], figsize=(15, 15))
    for i in range(params.shape[1]):
        for j in range(params.shape[1]):
            qs = np.quantile(params[:, i], [cut, 1 - cut])
            param_i = params[:, i][(qs[0] <= params[:, i]) &
                                   (params[:, i] <= qs[1])]
            if i == j:
                ax[i, j].hist(param_i, bins=bins)
            else:
                qs = np.quantile(params[:, j], [cut, 1 - cut])
                param_j = params[:, j][(qs[0] <= params[:, j]) &
                                       (params[:, j] <= qs[1])]
                ax[i, j].hist2d(param_j, param_i, bins=bins)
            if j % 2 == 0:
                plt.setp(ax[-1, j], xlabel=f'w{j + 1 - j // 2}')
            else:
                plt.setp(ax[-1, j], xlabel=f'D{j - j // 2}')
        if i % 2 == 0:
            plt.setp(ax[i, 0], ylabel=f'w{i + 1 - i // 2}')
        else:
            plt.setp(ax[i, 0], ylabel=f'D{i - i // 2}')
