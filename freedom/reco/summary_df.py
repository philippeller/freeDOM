"""
Provides functions for converting FreeDOM fit results into pandas dataframes
"""

__author__ = "Aaron Fienberg"

import pandas as pd


def build_summary_df(all_outs, par_names):
    """takes a list of fit outputs and builds a summary dataframe"""
    n_params = len(par_names)

    evt_idx = []
    free_fit_llhs = []
    true_param_llhs = []
    retro_param_llhs = []
    n_calls = []
    n_iters = []
    best_fit_ps = [[] for _ in range(n_params)]

    for i, out in enumerate(all_outs):
        freedom_params = out[0]["x"]
        freedom_llh = out[0]["fun"]
        n_calls.append(out[0]["n_calls"])
        n_iters.append(out[0]["nit"])

        evt_idx.append(i)
        free_fit_llhs.append(freedom_llh)
        true_param_llhs.append(out[1])
        retro_param_llhs.append(out[2])

        for p_ind, p in enumerate(freedom_params):
            best_fit_ps[p_ind].append(p)

    df_dict = dict(
        evt_idx=evt_idx,
        free_fit_llh=free_fit_llhs,
        true_p_llh=true_param_llhs,
        retro_p_llh=retro_param_llhs,
        n_calls=n_calls,
        n_iters=n_iters,
    )

    for p_name, p_list in zip(par_names, best_fit_ps):
        df_dict[p_name] = p_list

    return pd.DataFrame(df_dict)
