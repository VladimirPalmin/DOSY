from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.bootstrapping import bootstrap, final_guess
from src.mixture_fit import fits, fits_baes, sum_exp


def plot(x: np.ndarray,
         y: np.ndarray,
         title: Optional[str] = None,
         fontsize: int = 15) -> None:
    """
    Simple plot for experimental data.

    Parameters
    ----------
    x: np.ndarray
        X-values for the experiment. In original analysis it is
        normalized Z values.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    title: Optional[str] = None
        Title of the plot.
    fontsize: int = 15
        Font of the text in plot.
    """
    plt.scatter(x, y, color="red", s=10, label="data")
    plt.ylabel('$I/I_0$', fontsize=fontsize)
    plt.xlabel('Z * 1e-6', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)


def metrics_plot(aics: np.ndarray,
                 aic_probs: np.ndarray,
                 bics: np.ndarray,
                 bic_probs: np.ndarray) -> None:
    """
    Plot of the AIC, BIC probabilities.

    Parameters
    ----------
    aics: np.ndarray
        Array of absolute AIC-values for every parameter set in params.
    aic_probs: np.ndarray
        Probabilities of the parameter sets to be correct according
        to AIC analysis.
    bics: np.ndarray
        Array of absolute BIC-values for every parameter set in params.
    bic_probs: np.ndarray
        Probabilities of the parameter sets to be correct according
        to BIC analysis.
    """
    plt.subplot(121)
    plt.plot(range(1, len(aics) + 1), aic_probs, '.')
    plt.hlines(0.32, 1, len(aics) + 1, 'r', alpha=0.5)
    plt.hlines(0.05, 1, len(aics) + 1, 'r', alpha=0.5)
    plt.ylabel('exp($\Delta$AIC/2)')
    plt.xlabel('number of exponents')
    plt.title("AIC")

    plt.subplot(122)
    plt.plot(range(1, len(bics) + 1), bic_probs, '.')
    plt.hlines(0.32, 1, len(bics) + 1, 'r', alpha=0.5)
    plt.hlines(0.05, 1, len(bics) + 1, 'r', alpha=0.5)
    plt.ylabel('exp($\Delta$BIC/2)')
    plt.xlabel('number of exponents')
    plt.title("BIC")
    plt.show()


def print_params(num: int,
                 params: np.ndarray,
                 params_std: Optional[np.ndarray]) -> None:
    """
    Print the estimated parameters of the model.

    Parameters
    ----------
    num: int
        Number of components.
    params_opt: np.ndarray
        Array of parameters.
    params_opt_std: np.ndarray
        Array of std of parameters.
    """
    print(f'Number of components = {num}')
    if params_std is not None:
        for i in range(0, len(params)//2):
            print(f'W{i+1} = {params[2*i]:.3f} ± {params_std[2*i]:.3f}, '
                  f'D{i+1} = {params[2*i+1]:.3f} ± {params_std[2*i+1]:.3f}')
    else:
        for i in range(0, len(params)//2):
            print(f'W{i+1} = {params[2*i]:.3f}, '
                  f'D{i+1} = {params[2*i+1]:.3f}')


def analysis(x: np.ndarray,
             y: np.ndarray,
             conf_level: float = 2.0,
             bs_iters: int = 100,
             calc_sigma: float = 0.018,
             func: Callable = fits,
             *args, **kwargs) -> np.ndarray:
    """
    Performs analysis of given data with given estimator.

    Parameters
    ----------
    x: np.ndarray,
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray,
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    conf_level: float = 2.0
        Coefficint to define the level of confidence. E.g. level=2.0
        represents the 2-sigma confidence.
    bs_iters: int = 100
        Number of generated experiments to perform in boostrapping.
    calc_sigma: float = 0.018
        Estimation for the maximum noise level in the Y data.
        Typical value for the dataset used in the original analysis is 0.018.
    func: callable = fits
        The estimator to be used in the analysis.
        Note that it is expected to iterate through the several models
        for proper analysis. For reference see the 'fits' function in
        mixture_fit.py.
    *args, **kwargs
        Used to pass arguments to func callable. Avoid repetitions
        with analysis function arguments.

    Returns
    -------
    np.ndarray
        Array of results for different numbers of models components.
        Each element is a tuple with the following structure:
        (index of the paramaters,
        fraction of final guesses during bootstrapping,
        result of estimation on real data,
        mean result of estimation on bootstrapped data,
        std of estimation on bootstrapped data)
    """
    res = func(x, y, *args, **kwargs)
    y_model = sum_exp(res[-1], x)
    res_multi, res_opt = bootstrap(func, x, y_model, calc_sigma=calc_sigma,
                                   num=bs_iters, conf_level=conf_level,
                                   *args, **kwargs)

    output = []
    for i in range(len(res)):
        indx = res_opt[:, 0] == i
        prob = indx.mean()
        val, sigma = None, None
        if prob > 0:
            val = res_opt[:, 1][indx].mean()
            sigma = res_opt[:, 1][indx].std(0)
        output.append((i, prob, res[i], val, sigma))
    return np.vstack(output)


def data_analysis(x: np.ndarray,
                  y: np.ndarray,
                  bs_iters: int = 100,
                  method: str = 'BFGS',
                  reg: float = 1.3,
                  sigma: Optional[Union[float, np.ndarray]] = 0.02
                  ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Final estimation of the parameters of data with WLS estimator.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    bs_iters: int = 100
        Number of generated experiments to perform in boostrapping.
    method: str = 'BFGS'
        Method to choose for the optimization procedure. Must be on of the list
        {least_sqaures, curve_fit, dual_annealing, L-BFGS-B, BFGS}.
    reg: float = 1.3
        Weight-coefficient for the regularizaion component of loss function.
        Ignored in 'least_squares' and 'curve_fit'.
    sigma: float or nd.array or None = 0.02
        Expected deviation in experimental points.
        If float the array is generated with error_estimate with given
        maximum value.
        If None chi-square is reduced to simple least squares expression.
        Ignored in 'least_squares' and 'curve_fit'.

    Returns
    -------
    number: int
        Number of components in the optimal model.
    params_opt: np.ndarray
        Array of optimal parameters.
    params_opt_std: np.ndarray
        Array of std of optimal parameters calculated using boostrapping.
    """
    ress = analysis(x, y, bs_iters=bs_iters, method=method,
                    reg=reg, sigma=sigma, boost=True)
    params = []
    params_std = []
    for res in ress:
        params.append(res[2])
        params_std.append(res[4])
    rind = np.where([isinstance(a, np.ndarray) for a in params_std])[0]
    params_std = np.array(params_std)[rind]
    params = np.array(params)[rind]
    indx, params_opt, params_opt_std = final_guess(x, y, 0.02,
                                                   params, params_std)
    num = len(params_opt)//2
    return num, params_opt, params_opt_std


def data_analysis_baes(x: np.ndarray,
                       y: np.ndarray,
                       bs_iters: int = 100,
                       params_baes: np.ndarray = np.array([0.3, 1.06,
                                                           0.7, 0.46]),
                       params_baes_sigma: float = 0.1):
    """
    Final estimation of the parameters of data with MAP estimator.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    bs_iters: int = 100
        Number of generated experiments to perform in boostrapping.
    params_baes: np.ndarray
        Array with the self-diffusion coefficients expected in the data.
    params_baes_sigma: float
        The value is used as deviation for all self-diffusion coefficients.

    Returns
    -------
    num: int
        Number of components in the optimal model.
    params_opt: np.ndarray
        Array of optimal parameters.
    params_opt_std: np.ndarray
        Array of std of optimal parameters calculated using boostrapping.
    """
    res_map = fits_baes(x, y, sigma=0.018,
                        params_baes=params_baes,
                        params_baes_sigma=params_baes_sigma)
    ress = analysis(x, y, bs_iters=bs_iters,
                    func=fits_baes, mode='experiment', sigma=0.018,
                    params_baes=res_map, params_baes_sigma=params_baes_sigma)
    params = []
    params_std = []
    for res in ress:
        params.append(res[2])
        params_std.append(res[4])
    rind = np.where([isinstance(a, np.ndarray) for a in params_std])[0]
    params_std = np.array(params_std)[rind]
    params = np.array(params)[rind]
    indx, params_opt, params_opt_std = final_guess(x, y, 0.02,
                                                   params, params_std)
    num = len(params_opt)//2
    return num, params_opt, params_opt_std
