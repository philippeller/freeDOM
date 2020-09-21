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
