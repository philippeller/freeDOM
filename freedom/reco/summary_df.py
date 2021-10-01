"""
Provides functions for converting FreeDOM fit results into pandas dataframes
"""


import pandas as pd


def build_summary_df(all_outs, par_names):
    """takes a list of fit outputs and builds a summary dataframe"""
    n_params = len(par_names)

    evt_idx = []
    free_fit_llhs = []
    true_param_llhs = []
    retro_param_llhs = []
    reco_times = []
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
        reco_times.append(out[2])
        if len(out) > 3:
            retro_param_llhs.append(out[3])

        for p_ind, p in enumerate(freedom_params):
            best_fit_ps[p_ind].append(p)

    df_dict = dict(
        evt_idx=evt_idx,
        free_fit_llh=free_fit_llhs,
        true_p_llh=true_param_llhs,
        reco_time=reco_times,
        n_calls=n_calls,
        n_iters=n_iters,
    )
    if len(retro_param_llhs) != 0:
        df_dict["retro_p_llh"] = retro_param_llhs

    for p_name, p_list in zip(par_names, best_fit_ps):
        df_dict[p_name] = p_list

    try:
        add_postfit_res(df_dict, all_outs, par_names)
    except KeyError:
        pass

    return pd.DataFrame(df_dict)


def add_postfit_res(df_dict, all_outs, par_names):
    mean_ests = [[] for _ in par_names]
    env_ests = [[] for _ in par_names]
    curvatures = [[] for _ in par_names]
    stds = [[] for _ in par_names]
    hull_areas = [[] for _ in par_names]
    furthest_points = [[] for _ in par_names]

    for out in all_outs:
        pf = out[0]["postfit"]
        for i, _ in enumerate(par_names):
            mean_ests[i].append(pf["means"][i])
            env_ests[i].append(pf["env_mins"][i])
            curvatures[i].append(pf["envs"][i][2])
            stds[i].append(pf["stds"][i])
            hull_areas[i].append(pf["hull_areas"][i])
            furthest_points[i].append(pf["furthest_points"][i])

    for i, name in enumerate(par_names):
        df_dict[f"{name}_mean"] = mean_ests[i]
        df_dict[f"{name}_env_est"] = env_ests[i]
        df_dict[f"{name}_curvature"] = curvatures[i]
        df_dict[f"{name}_std"] = stds[i]
        df_dict[f"{name}_hull_area"] = hull_areas[i]
        df_dict[f"{name}_furthest_point"] = furthest_points[i]
