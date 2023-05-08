from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import numpy as np

from src.mixture_fit import error_estimate
from src.optimal_number import optimal_params


def conf_intervals(params, sigmas, level=2):
    intervals = np.zeros((len(params), 2), dtype=object)
    for i in range(len(params)):
        intervals[i] = params[i] - level * sigmas[i], params[i] + level * sigmas[i]
    return intervals


def check_similarity(theta, intervals):
    for param in theta[1::2]:
        entries = np.sum([1 if interval[0] < param < interval[1] else 0 for interval in intervals[1::2]])
        if entries > 1:
            return True
    return False


def final_guess(x, y, sigma, params, params_std, conf_level=2):
    aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_idx = optimal_params(x, y, params, sigma)

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
        check_zero = np.any(intervals.T[0][1::2] < 0)
        check = check_sim or check_negative or check_zero
    return indx, params_opt, params_opt_std


def bootstrap(function, x, y_model, calc_sigma, num=100, conf_level=2, show_progress=True, *args, **kwargs):
    est_coeff = -(np.log(y_model[0]) - np.log(y_model[-1]))/(x[0] - x[-1])
    calc_sigma = error_estimate(x, calc_sigma, est_coeff)
    # calc_sigma = calc_sigma * y_model/y_model.max()

    with ProcessPoolExecutor(max_workers=10) as pool:
        samples = list(map(lambda _: (x, y_model + np.random.normal(0, calc_sigma, len(x))), range(num)))

        futures = [pool.submit(function, x1, y1, *args, **kwargs) for x1, y1 in samples]

        results = []
        for future in tqdm(as_completed(futures), total=num, disable=(not show_progress)):
            results.append(future.result())
        results = np.vstack(results)
        params_std = ((results - results.mean(0)) ** 2).mean(0) ** 0.5

        optim_futures = [pool.submit(final_guess, x1, y1, calc_sigma, params, params_std, conf_level=conf_level)
                         for (x1, y1), params in zip(samples, results)]
        optim_results = []
        for future in as_completed(optim_futures):
            optim_results.append(future.result())
        optim_results = np.vstack(optim_results)

        return results, optim_results
