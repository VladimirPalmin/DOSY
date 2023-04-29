import matplotlib.pyplot as plt
import numpy as np

from src.errors import bootstrap_resudial
from src.mixture_fit import fits, sum_exp
from src.optimal_number import optimal_params
from src.bootstrapping import bootstrap


def plot(x, y, title=None, fontsize=15):
    plt.scatter(x, y, color="red", s=10, label="data")
    plt.ylabel('$I/I_0$', fontsize=fontsize)
    plt.xlabel('Z * 1e-6', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)


def param_print(array):
    print("(w, D)")
    print("-----------------")
    for a in array:
        for i in range(0, len(a), 2):
            print((a[i], a[i + 1]))
        print()


def metrics_plot(aics, aic_probs, bics, bic_probs):
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


def conf_intervals(params, sigmas, level=2):
    intervals = np.zeros((len(params), 2), dtype=object)
    for i in range(len(params)):
        intervals[i] = params[i] - level * sigmas[i], params[i] + level * sigmas[i]
    return intervals


def estimate_sigmas(thetas):
    return np.std(thetas, axis=0)


def check_similarity(theta, intervals):
    for param in theta[1::2]:
        entries = np.sum([1 if interval[0] < param < interval[1] else 0 for interval in intervals[1::2]])
        if entries > 1:
            return True
    return False


def check_zero(theta, intervals):
    entries = np.sum([1 if (interval[0] < 0)  else 0 for interval in intervals[1::2]])
    return (entries == 0)


def check_negative(params):
    check = [np.all(theta > 0) for theta in params]
    indx = np.arange(len(params))
    return indx[check][-1] + 1


def number_analysis(x, y, n_min=1, n_max=3, method="BFGS", reg=0.005, plot_print=False):
    params = fits(x, y, n_min, n_max, method, reg)
    aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_number = optimal_params(x, y, params)
    if plot_print:
        metrics_plot(aics, aic_probs, bics, bic_probs)
        print(f"{method}")
        print("---------------------------")
        param_print(params)
        print("---------------------------")
        print(f"AIC: {m_aic + 1}")
        print(f"BIC: {m_bic + 1}")
        print(f"conservative: {cons_number + 1}")

    return params, m_aic + 1, m_bic + 1, cons_number + 1


def error_analysis(n, x, y, method='BFGS', reg=0.0,
                   bs_iters=1000, bs_method='residuals', seed=42):
    init_theta, thetas, res = bootstrap_resudial(n, x, y, bs_iters, bs_method,
                                                 method, seed, reg)
    return init_theta, thetas, res


def analysis(x, y, conf_level=2, bs_iters=1000, calc_sigma=0.02, func=fits, *args, **kwargs):
    res = func(x, y, *args, **kwargs)
    y_model = sum_exp(res[-1], x)
    res_multi, res_opt = bootstrap(func, x, y_model, calc_sigma=calc_sigma, num=bs_iters, conf_level=conf_level, *args, **kwargs)

    output = []
    for i in range(len(res)):
        indx = res_opt[:, 0] == i
        prob = indx.mean()
        val, sigma = None, None
        if prob > 0:
            val = res_opt[:, 1][indx].mean()
            sigma = res_opt[:, 1][indx].std(0)
        output.append((i, prob, res[i], val, sigma))
    return output, res_multi


#THIS CODE IS DEPRECATED, PLEASE DO NOT USE IT. CHECK bootstrapping.py final_guess() INSTEAD
def data_analysis(x, y, n_min=1, n_max=3, method="BFGS", reg=0.005, 
                  conf_level=2, bs_iters=1000, bs_method='residuals', seed=42):
    params, m_aic, m_bic, cons_number = number_analysis(x, y, n_min=n_min, n_max=n_max,
                                                        method=method, reg=reg)
    params_opt = params[:cons_number]
    indx = check_negative(params_opt)
    theta_opt = params_opt[indx - 1]
    init_theta, thetas, res = error_analysis(n=indx, x=x, y=y, method=method, reg=reg,
                                             bs_iters=bs_iters, bs_method=bs_method, seed=seed)
    sigmas = estimate_sigmas(thetas)
    intervals = conf_intervals(theta_opt, sigmas, conf_level)
    check = check_similarity(theta_opt, intervals)
    check = np.logical_or(check, check_zero(theta_opt, intervals))
    while check or indx > 1:
        init_theta, thetas, res = error_analysis(n=indx, x=x, y=y, method=method, reg=reg,
                                                 bs_iters=bs_iters, bs_method=bs_method, seed=seed)
        sigmas = estimate_sigmas(thetas)
        intervals = conf_intervals(theta_opt, sigmas)
        check = check_similarity(theta_opt, intervals)
        check = np.logical_or(check, check_zero(theta_opt, intervals))
        if check:
            indx = indx - 1
            theta_opt = params_opt[indx - 1]
        else:
            return indx, theta_opt, sigmas, params
    return indx, theta_opt, sigmas, params
