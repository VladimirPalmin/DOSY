import numpy as np

from src.mixture_fit import sum_exp_curv


def chi_square(y, y_pred, sigma):
    dof = len(y)
    return np.sum((y - y_pred) ** 2 / sigma ** 2) / dof


def aic(y, y_pred, params, sigma):
    n = len(y)
    k = len(params)
    return n * np.log(chi_square(y, y_pred, sigma)) + 2 * k


def bic(y, y_pred, params, sigma):
    n = len(y)
    k = len(params)
    return n * np.log(chi_square(y, y_pred, sigma)) + k * np.log(n)


def AIC_analysis(x, y, params, sigma):
    n = len(params)
    aics = np.zeros(n)
    for i in range(n):
        y_pred = sum_exp_curv(x, *params[i])
        aics[i] = aic(y, y_pred, params[i], sigma)
    min_aic_number = np.argmin(aics)
    min_aic = aics[min_aic_number]
    probs = np.exp((min_aic - aics) / 2)
    return min_aic_number, aics, probs


def BIC_analysis(x, y, params, sigma):
    n = len(params)
    bics = np.zeros(n)
    for i in range(n):
        y_pred = sum_exp_curv(x, *params[i])
        bics[i] = bic(y, y_pred, params[i], sigma)
    min_bic_number = np.argmin(bics)
    min_bic = bics[min_bic_number]
    probs = np.exp((min_bic - bics) / 2)
    return min_bic_number, bics, probs


def chi2_analysis(x, y, params, sigma, chi2_rel_change=0.01):
    n = len(params)
    chi2 = np.zeros(n)
    for i in range(n):
        y_pred = sum_exp_curv(x, *params[i])
        chi2[i] = chi_square(y, y_pred, sigma)
    chi2_number = np.where(-np.diff(chi2) / chi2[1:] < chi2_rel_change)[0][0]
    return chi2_number, chi2


def optimal_params(x, y, params, sigma=1):
    m_aic, aics, aic_probs = AIC_analysis(x, y, params, sigma)
    m_bic, bics, bic_probs = BIC_analysis(x, y, params, sigma)
    try:
        cons_number = np.where(bic_probs > 0.05)[0][0]
    except IndexError:
        cons_number = m_bic
    return aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_number
