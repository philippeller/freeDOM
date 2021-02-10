"""
Provides "postfit" functions. Postfits analyze the minimizer samples 
to provide better parameter estimates 
"""

__author__ = "Aaron Fienberg"

import numpy as np

DELTA_LLH_CUT = 15
DEFAULT_LOC_SPACING = (-1.5, 1.5, 10)
PAR_NAMES = [
    "x",
    "y",
    "z",
    "time",
    "azimuth",
    "zenith",
    "cascade energy",
    "track energy",
]


def get_stats(all_pts, par_names):
    llhs = all_pts[:, -1]
    w = np.exp(llhs.min() - llhs)

    means = np.zeros(len(par_names))
    variances = np.zeros(len(par_names))
    for i, name in enumerate(par_names):
        if "zenith" not in name and "azimuth" not in name:
            mean = np.average(all_pts[:, i], weights=w)
            means[i] = mean
            variances[i] = np.average((all_pts[:, i] - mean) ** 2, weights=w)

    zen_ind = par_names.index("zenith")
    az_ind = par_names.index("azimuth")

    zen = all_pts[:, zen_ind]
    az = all_pts[:, az_ind]

    p_x = np.average(np.sin(zen) * np.cos(az), weights=w)
    p_y = np.average(np.sin(zen) * np.sin(az), weights=w)
    p_z = np.average(np.cos(zen), weights=w)

    r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)

    means[zen_ind] = np.arccos(p_z / r)
    means[az_ind] = np.arctan2(p_y, p_x) % (2 * np.pi)

    variances[zen_ind] = np.average(
        (all_pts[:, zen_ind] - means[zen_ind]) ** 2, weights=w
    )
    variances[az_ind] = np.average((all_pts[:, az_ind] - means[az_ind]) ** 2, weights=w)

    return means, variances


def get_envelope(
    par, llhs, mean, std, start_step=0.05, loc_spacing=DEFAULT_LOC_SPACING
):
    min_llh = llhs.min()

    locs = np.linspace(*loc_spacing)
    xs = []
    ys = []
    for loc in locs:
        bin_cent = mean + std * loc

        step = start_step
        while True:
            pt_q = (par > bin_cent - step * std) & (par < bin_cent + step * std)
            if np.any(pt_q):
                break
            else:
                step = step * 2

        loc_llhs = llhs[pt_q]
        loc_pars = par[pt_q]

        min_ind = np.argmin(loc_llhs)
        xs.append(loc_pars[min_ind])
        ys.append(loc_llhs[min_ind] - min_llh)

    return np.polyfit(xs, ys, 2), xs, ys


def get_parabola_min(quad_coeffs):
    return -quad_coeffs[1] / (2 * quad_coeffs[0])


def postfit(all_pts, par_names=PAR_NAMES, llh_cut=DELTA_LLH_CUT):
    all_pts = np.asarray(all_pts)
    all_llhs = all_pts[:, -1]
    cut_pts = all_pts[all_llhs < all_llhs.min() + llh_cut]
    cut_llhs = cut_pts[:, -1]

    means, variances = get_stats(cut_pts, par_names)
    stds = np.sqrt(variances)

    env_rets = [
        get_envelope(par, cut_llhs, mean, std)
        for par, mean, std in zip(cut_pts[:, :-1].T, means, stds)
    ]
    envs, env_xs, env_ys = list(zip(*env_rets))

    env_mins = [get_parabola_min(env) for env in envs]

    return dict(
        means=means,
        stds=stds,
        envs=envs,
        env_mins=env_mins,
        #         env_xs=env_xs,
        #         env_ys=env_ys,
    )
