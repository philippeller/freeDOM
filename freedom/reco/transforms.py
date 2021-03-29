"""
Parameter transformations for reparameterizing the LLH function

transforms are from fitter parameterization to network parameterization
inverse transforms are from network parameterization to fitter parameterization 
"""
import functools

import numpy as np

from freedom.utils.i3frame_dataloader import DEFAULT_LABELS

DEFAULT_CASC_E_IND = DEFAULT_LABELS.index("cascade_energy")
DEFAULT_TRACK_E_IND = DEFAULT_LABELS.index("track_energy")


def param_transform(f):
    """handles 1d and 2d parameter arrays; 
    ensures params are copied before transformation"""

    @functools.wraps(f)
    def trans(params, *args, **kwargs):
        params = np.atleast_1d(params)

        orig_shape = params.shape
        if len(orig_shape) == 1:
            params = params[np.newaxis, :]

        trans_params = f(np.copy(params), *args, **kwargs)

        return trans_params.reshape(orig_shape)

    return trans


@param_transform
def track_fraction_transform(
    params, casc_E_ind=DEFAULT_CASC_E_IND, track_E_ind=DEFAULT_TRACK_E_IND
):
    """from total_E, track_frac to casc_E, track_E"""
    total_E = params[:, casc_E_ind]
    track_frac = params[:, track_E_ind]

    params[:, track_E_ind] = track_frac * total_E
    params[:, casc_E_ind] = total_E - params[:, track_E_ind]

    return params


@param_transform
def inv_track_fraction_transform(
    params, casc_E_ind=DEFAULT_CASC_E_IND, track_E_ind=DEFAULT_TRACK_E_IND
):
    """from casc_E, track_E to total_E, track_frac"""
    total_E = params[:, [casc_E_ind, track_E_ind]].sum(axis=1)
    track_frac = params[:, track_E_ind] / total_E

    params[:, casc_E_ind] = total_E
    params[:, track_E_ind] = track_frac

    return params


track_frac_names = DEFAULT_LABELS[:6] + ["total_energy", "track_frac"]

track_frac_transforms = dict(
    trans=track_fraction_transform,
    inv_trans=inv_track_fraction_transform,
    par_names=track_frac_names,
)
