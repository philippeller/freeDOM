"""
Provides "postfit" functions. Postfits analyze the minimizer samples
to provide alternative parameter estimates and uncertainty estimates
"""

__author__ = "Aaron Fienberg"

import numpy as np
import numpy.polynomial.polynomial as poly

DELTA_LLH_CUT = 15
DEFAULT_LOC_SPACING = (-1.5, 1.5, 10)
DEFAULT_START_STEP = 0.05
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


def get_stats(all_pts, par_names, do_angles=True):
    """Calculate likelihood-weighted mean and variance

    Given llh samples from an optimizer, this function calculates the likelihood-weighted
    mean and variance for each parameter

    Parameters
    ----------
    all_pts : np.ndarray
        the optimizer samples; the negative llh values should be in the final column
    par_names : list
        the parameter names
    do_angles : bool, default True
        whether to calculate stats for the azimuth, zenith parameters

    Returns
    -------
    tuple
       ([means], [variances])
    """
    llhs = all_pts[:, -1]
    w = np.exp(llhs.min() - llhs)

    means = np.zeros(len(par_names))
    variances = np.zeros(len(par_names))
    for i, name in enumerate(par_names):
        if "zenith" not in name and "azimuth" not in name:
            mean = np.average(all_pts[:, i], weights=w)
            means[i] = mean
            variances[i] = np.average((all_pts[:, i] - mean) ** 2, weights=w)

    try:
        zen_ind = par_names.index("zenith")
        az_ind = par_names.index("azimuth")
    except ValueError:
        do_angles = False

    if do_angles:
        zen = all_pts[:, zen_ind]
        az = all_pts[:, az_ind]

        p_x = np.average(np.sin(zen) * np.cos(az), weights=w)
        p_y = np.average(np.sin(zen) * np.sin(az), weights=w)
        p_z = np.average(np.cos(zen), weights=w)

        r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)

        if r != 0:
            means[zen_ind] = np.arccos(p_z / r)
            means[az_ind] = np.arctan2(p_y, p_x) % (2 * np.pi)

            variances[zen_ind] = np.average(
                (all_pts[:, zen_ind] - means[zen_ind]) ** 2, weights=w
            )
            variances[az_ind] = np.average(
                (all_pts[:, az_ind] - means[az_ind]) ** 2, weights=w
            )

    return means, variances


def get_envelope(
    par, llhs, mean, std, start_step=DEFAULT_START_STEP, loc_spacing=DEFAULT_LOC_SPACING
):
    """Estimate 1d parabolic llh envelopes for a single parameter

    Parameters
    ----------
    par : np.ndarray
        the parameter values
    llhs : np.ndarray
        the llh values
    mean : float
        estimate of the parameter best fit value
    std : float
        estimate of the parameter uncertainty

    Returns
    -------
    tuple
       ([parabola coeffs], [fit x pts], [fit y pts])
    """
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

    return poly.polyfit(xs, ys, 2), xs, ys


def get_parabola_opt(quad_coeffs):
    """x value of the parabola optimum

    Parameters
    ----------
    quad_coeffs: np.ndarray
        the parabola coefficients; p0 + p1*x + p2*x^2 -> [p0, p1, p2]

    Returns
    -------
    float
    """
    if quad_coeffs[2] != 0:
        return -quad_coeffs[1] / (2 * quad_coeffs[2])
    else:
        return np.nan


def postfit(all_pts, par_names=PAR_NAMES, llh_cut=DELTA_LLH_CUT):
    """postfit routine for event reconstruction

    The postfit includes uncertainty estimation and alterative parameter estimators

    Parameters
    ----------
    all_pts: np.ndarray
        the optimizer samples; the negative llh values should be in the final column
    par_names : list, optional
        parameter names, defaults to
        ["x", "y", "z", "time", "azimuth", "zenith", "cascade energy", "track energy"]
        "azimuth" and "zenith" parameters receive special treatment
    llh_cut: float, default 15
        postfit routines only consider samples within llh_cut llh of the best sample

    Returns
    -------
    dict
    """
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

    env_mins = [get_parabola_opt(env) for env in envs]

    return dict(
        means=means,
        stds=stds,
        envs=envs,
        env_mins=env_mins,
        #         env_xs=env_xs,
        #         env_ys=env_ys,
    )
