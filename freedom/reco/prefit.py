"""
Provides "prefit" functions. Prefits give initial guesses that can be used
to seed llh optimizers and samplers
"""

__author__ = "Aaron Fienberg"

import numpy as np


def initial_box(hits, init_range, charge_ind=4, n_params=8, pos_seed="CoG"):
    """ returns initial box limits for each dimension
    in the form of a n_params x 2 table
    
    returned energy limits are in units of log energy
    """

    limits = np.empty((n_params, 2), np.float32)

    if pos_seed == "CoG":
        # charge weighted positions, time
        hit_avgs = np.average(hits, weights=hits[:, charge_ind], axis=0)[:4]

        # x, y, z, t range from average + init_range[0] to average + init_range[1]
        limits[:4] = hit_avgs[:4, np.newaxis] + init_range[:4]
    else:
        raise ValueError(
            f'Only "CoG" based position seeds are currently supported. You selected {pos_seed}'
        )

    # angles and energies just span the specified ranges
    # (although the energy parameters are log energies)
    limits[4:] = init_range[4:]

    return limits


def truth_seed_box(true_params, init_range, az_ind=4, zen_ind=5):
    """generate initial box limits from the true params
    """
    n_params = len(true_params)
    true_params = np.copy(true_params[:, np.newaxis])
    # clip true energies between 0.3 GeV and 1000 GeV
    true_params[-2:] = true_params[-2:].clip(0.3, 1000)

    limits = np.empty((n_params, 2), np.float32)

    limits[:-2] = true_params[:-2] + init_range[:-2]

    limits[-2:] = np.log10(true_params[-2:]) + init_range[-2:]

    limits[az_ind] = limits[az_ind].clip(0, 2 * np.pi)
    limits[zen_ind] = limits[zen_ind].clip(0, np.pi)

    return limits
