"""
Provides functions for parameter bounds checking during LLH evaluations
"""

__author__ = "Aaron Fienberg"

import numpy as np

NAN_REPLACE_VAL = 1e10


def get_out_of_bounds_func(limits, bounds_check_type="cube"):
    """returns func returning a boolean array, True for param rows that are out of bounds"""

    if bounds_check_type == "cube":

        def out_of_bounds(params):
            """ "cube" bounds_check_type; checks each parameter independently"""
            return ~np.alltrue(
                np.logical_and(limits[0] <= params, params <= limits[1]), axis=-1
            )

    else:
        raise ValueError(
            f'Only "cube" bounds checks are currently supported; You selected {bounds_check_type}'
        )

    return out_of_bounds


def nan_replace(llhs):
    """replaces nans with large, valid values"""
    llhs[np.isnan(llhs)] = NAN_REPLACE_VAL
    return llhs


def out_of_bounds_replace(llhs, params, out_of_bounds):
    """ replace out of bounds llh evals with large, valid values"""
    llhs[out_of_bounds(params)] = NAN_REPLACE_VAL
    return llhs


def invalid_replace(llhs, params, out_of_bounds):
    """combines out of bounds replace and nan replace"""
    return out_of_bounds_replace(nan_replace(llhs), params, out_of_bounds_replace)
