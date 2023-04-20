from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import numpy as np

from src.optimal_number import optimal_params
from src.main import check_similarity, conf_intervals
from src.mixture_fit import error_estimate


def final_guess(x, y, sigma, params, params_std, conf_level=2):
    aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_idx = optimal_params(x, y, params, sigma)

    params_opt = params[cons_idx]
    params_opt_std = params_std[cons_idx]
    indx = cons_idx

    intervals = conf_intervals(params_opt, params_opt_std, conf_level)
    check = np.logical_or(check_similarity(params_opt, intervals), np.any(params_opt < 0))
    while check and indx > 0:
        indx = indx - 1
        params_opt = params[indx]
        params_opt_std = params_std[indx]
        intervals = conf_intervals(params_opt, params_opt_std, conf_level)
        check = np.logical_or(check_similarity(params_opt, intervals), np.any(params_opt < 0))
    return indx, params_opt, params_std


def bootstrap(function, x, y_model, calc_sigma, num=100, show_progress=True, *args, **kwargs):
    calc_sigma = error_estimate(x, calc_sigma)

    with ProcessPoolExecutor(max_workers=10) as pool:
        samples = list(map(lambda _: (x, y_model + np.random.normal(0, calc_sigma, len(x))), range(num)))

        futures = [pool.submit(function, x1, y1, *args, **kwargs) for x1, y1 in samples]

        results = []
        for future in tqdm(as_completed(futures), total=num, disable=(not show_progress)):
            results.append(future.result())
        results = np.vstack(results)
        params_std = ((results - results.mean(0)) ** 2).mean(0) ** 0.5

        optim_futures = [pool.submit(final_guess, x1, y1, calc_sigma, params, params_std)
                  for (x1, y1), params in zip(samples, results)]
        optim_results = []
        for future in as_completed(optim_futures):
            optim_results.append(future.result())
        optim_results = np.vstack(optim_results)

        return results, optim_results
