import numpy as np

from src.mixture_fit import sum_exp, chi_square

from typing import Union, Tuple


def aic(y: np.ndarray,
        y_pred: np.ndarray,
        params: np.ndarray,
        sigma: Union[np.ndarray, float]) -> float:
    """
    Calculation of absolute AIC metric for given data.

    Parameters
    ----------
    y: np.ndarray
        Experimental data for analysis.
    y_pred: np.ndarray
        Modelled data for analysis. In original work is is
        likely a sum_exp(params, x).
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    sigma: np.ndarray or float
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
    float
        AIC-metric value for given data and model parameters.
    """
    n = len(y)
    k = len(params)
    return n * np.log(chi_square(y, y_pred, sigma)) + 2 * k


def bic(y: np.ndarray,
        y_pred: np.ndarray,
        params: np.ndarray,
        sigma: Union[np.ndarray, float]) -> float:
    """
    Calculation of absolute BIC metric for given data.

    Parameters
    ----------
    y: np.ndarray
        Experimental data for analysis.
    y_pred: np.ndarray
        Modelled data for analysis. In original work is is
        likely a sum_exp(params, x).
    params: np.ndarray
        Parameters of the model. Even params (0, 2, ...) represent weights
        of exponents. Odd params (1, 3, ...) - self-diffusion coefficients.
    sigma: np.ndarray or float
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
    float
        BIC-metric value for given data and model parameters.
    """
    n = len(y)
    k = len(params)
    return n * np.log(chi_square(y, y_pred, sigma)) + k * np.log(n)


def AIC_analysis(x: np.ndarray,
                 y: np.ndarray,
                 params: np.ndarray,
                 sigma: Union[np.ndarray, float]
                 ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Performs AIC analysis for the given data. Returns the index of
    optimal parameters. The optimal number of components equals
    (optimal index + 1).

    Parameters
    ----------
    y: np.ndarray
        Experimental data for analysis.
    y_pred: np.ndarray
        Modelled data for analysis. In original work is is
        likely a sum_exp(params, x).
    params: np.ndarray
        Array with the np.ndarray elements. i-th element shows the estimated
        parameters for the model with (i+1) components. In each element of list
        even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    sigma: np.ndarray or float
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
    min_aic_number: int
        Index of the optimal parameters in params.
    aics: np.ndarray
        Array of absolute AIC-values for every parameter set in params.
    probs: np.ndarray
        Probabilities of the parameter sets to be correct according
        to AIC analysis.
    """
    n = len(params)
    aics = np.zeros(n)
    for i in range(n):
        y_pred = sum_exp(params[i], x)
        aics[i] = aic(y, y_pred, params[i], sigma)
    min_aic_number = np.argmin(aics)
    min_aic = aics[min_aic_number]
    probs = np.exp((min_aic - aics) / 2)
    return min_aic_number, aics, probs


def BIC_analysis(x: np.ndarray,
                 y: np.ndarray,
                 params: np.ndarray,
                 sigma: Union[np.ndarray, float]
                 ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Performs BIC analysis for the given data. Returns the index of
    optimal parameters. The optimal number of components equals
    (optimal index + 1).

    Parameters
    ----------
    y: np.ndarray
        Experimental data for analysis.
    y_pred: np.ndarray
        Modelled data for analysis. In original work is is
        likely a sum_exp(params, x).
    params: np.ndarray
        Array with the np.ndarray elements. i-th element shows the estimated
        parameters for the model with (i+1) components. In each element of list
        even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    sigma: np.ndarray or float
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
    min_bic_number: int
        Index of the optimal parameters in params.
    bics: np.ndarray
        Array of absolute BIC-values for every parameter set in params.
    probs: np.ndarray
        Probabilities of the parameter sets to be correct according
        to BIC analysis.
    """
    n = len(params)
    bics = np.zeros(n)
    for i in range(n):
        y_pred = sum_exp(params[i], x)
        bics[i] = bic(y, y_pred, params[i], sigma)
    min_bic_number = np.argmin(bics)
    min_bic = bics[min_bic_number]
    probs = np.exp((min_bic - bics) / 2)
    return min_bic_number, bics, probs


def optimal_params(x: np.ndarray,
                   y: np.ndarray,
                   params: np.ndarray,
                   sigma: Union[np.ndarray, float] = 1.0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              int, int, int]:
    """
    Performs both AIC and BIC analysis and returns the conservative
    estimation of optimal number of components.

    Parameters
    ----------
    x: np.ndarray
        X-values to calcuate the normalized I values in. Represent the
        normalized Z values in experiments.
    y: np.ndarray
        Y-values for the experiment. In original analysis it is
        normalized I values for the experiment.
    params: np.ndarray
        Array with the np.ndarray elements. i-th element shows the estimated
        parameters for the model with (i+1) components. In each element of list
        even params (0, 2, ...) represent weights of exponents.
        Odd params (1, 3, ...) - self-diffusion coefficients.
    sigma: np.ndarray or float = 1.0
        Expected deviation in experimental points.
        If float same sigma is used for every point.

    Returns
    -------
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
    m_aic: int
        Index of the optimal parameters in params according to AIC analysis.
    m_bic: int
        Index of the optimal parameters in params according to BIC analysis.
    cons_number: int
        Conservative estimation of the index of optimal parameters.
        Equals to the index of first bic_probs value exceeding 5%.
    """
    m_aic, aics, aic_probs = AIC_analysis(x, y, params, sigma)
    m_bic, bics, bic_probs = BIC_analysis(x, y, params, sigma)
    try:
        cons_number = np.where(bic_probs > 0.05)[0][0]
    except IndexError:
        cons_number = m_bic
    return aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_number
