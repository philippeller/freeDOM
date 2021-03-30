"""Provides I3Module(s) for FreeDOM reco"""

from freedom.reco.crs_reco import (
    batch_crs_fit,
    DEFAULT_SEARCH_LIMITS,
    DEFAULT_INIT_RANGE,
)
from freedom.utils import i3frame_dataloader
from freedom.llh_service.llh_client import LLHClient
import numpy as np
import math
import time

DEFAULT_N_LIVE_POINTS = 97
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_ITER = 10000
DEFAULT_SPHERICAL_INDICES = [[4, 5]]

TRACK_M_PER_GEV = 15 / 3.3
"""Borrowed from Retro's muon_hypo.py"""


class I3FreeDOMClient:
    """FreeDOM client IceTray module. Connects to a running LLHService"""

    def __init__(self, ctrl_addr, conf_timeout, rng=None):
        """initialize FreeDOM client, connect to LLH service"""
        self._llh_client = LLHClient(ctrl_addr, conf_timeout)

        if rng is None:
            self._rng = np.random.default_rng(None)
        else:
            self._rng = rng

    def __call__(
        self,
        frame,
        geo,
        reco_pulse_series_name,
        suffix="",
        init_range=DEFAULT_INIT_RANGE,
        search_limits=DEFAULT_SEARCH_LIMITS,
        n_live_points=DEFAULT_N_LIVE_POINTS,
        do_postfit=True,
        store_all=False,
        truth_seed=False,
        batch_size=DEFAULT_BATCH_SIZE,
    ):
        """reconstruct an event stored in an i3frame"""
        start = time.time()

        event = i3frame_dataloader.load_event(frame, geo, reco_pulse_series_name)

        fit_res = batch_crs_fit(
            event,
            clients=[self._llh_client],
            rng=self._rng,
            init_range=init_range,
            search_limits=search_limits,
            n_live_points=n_live_points,
            do_postfit=do_postfit,
            store_all=store_all,
            truth_seed=truth_seed,
            batch_size=DEFAULT_BATCH_SIZE,
            spherical_indices=DEFAULT_SPHERICAL_INDICES,
            max_iter=DEFAULT_MAX_ITER,
        )

        if event["params"] is not None:
            fit_res["truth_LLH"] = self._llh_client.eval_llh(
                event["hit_data"][0], event["evt_data"][0], event["params"]
            )

        delta = time.time() - start
        fit_res["delta"] = delta

        store_reco_output(frame, suffix, fit_res)


def store_reco_output(frame, suffix, fit_res):
    """store reco output in an i3frame"""
    from icecube.dataclasses import I3VectorString, I3VectorDouble, I3Double
    from icecube.icetray import I3Int, I3Bool

    prefix = f"FreeDOM_{suffix}_"

    par_names = i3frame_dataloader.DEFAULT_LABELS
    frame[f"{prefix}par_names"] = to_i3_vec(par_names, I3VectorString)

    frame[f"{prefix}best_fit"] = to_i3_vec(fit_res["x"], I3VectorDouble)
    frame[f"{prefix}success"] = I3Bool(fit_res["success"])

    for double_p, double_frame_name in [("fun", "best_LLH"), ("delta", "delta_T")]:
        frame[f"{prefix}{double_frame_name}"] = I3Double(float(fit_res[double_p]))

    try:
        frame[f"{prefix}truth_LLH"] = I3Double(float(fit_res["truth_LLH"]))
    except KeyError:
        pass

    for int_p, int_frame_name in [
        ("n_calls", "n_llh_calls"),
        ("nit", "n_crs_iters"),
        ("stopping_flag", "stopping_flag"),
    ]:
        frame[f"{prefix}{int_frame_name}"] = I3Int(fit_res[int_p])

    postfit_res = fit_res.get("postfit", {})

    for key, val in postfit_res.items():
        if key == "envs":
            for i, env_ps in enumerate(zip(*val)):
                frame[f"{prefix}env_p{i}"] = to_i3_vec(env_ps, I3VectorDouble)
        else:
            frame[f"{prefix}{key}"] = to_i3_vec(val, I3VectorDouble)

    reco_pars = {name: val for name, val in zip(par_names, fit_res["x"])}
    reco_pars["success"] = fit_res["success"]
    for particle_type in ("neutrino", "cascade", "track"):
        frame[f"{prefix}{particle_type}"] = build_i3_particle(reco_pars, particle_type)


def to_i3_vec(array, i3_vec_type):
    """convert a list/array to an I3Vec"""
    i3_vec = i3_vec_type()
    for val in array:
        i3_vec.append(val)
    return i3_vec


_energy_getters = dict(
    cascade=lambda pars: pars["cascade_energy"],
    track=lambda pars: pars["track_energy"],
    neutrino=lambda pars: pars["track_energy"] + pars["cascade_energy"],
)


def build_i3_particle(reco_pars, particle_type):
    """build an I3Particle from reco parameters"""
    from icecube.dataclasses import I3Particle, I3Constants, I3Position, I3Direction
    from icecube.icetray import I3Units

    shape_map = dict(
        cascade=I3Particle.ParticleShape.Cascade,
        track=I3Particle.ParticleShape.ContainedTrack,
        neutrino=I3Particle.ParticleShape.Primary,
    )

    particle = I3Particle()

    if reco_pars["success"]:
        particle.fit_status = I3Particle.FitStatus.OK
    else:
        particle.fit_status = I3Particle.GeneralFailure

    particle.dir = I3Direction(reco_pars["zenith"], reco_pars["azimuth"])
    particle.energy = _energy_getters[particle_type](reco_pars) * I3Units.GeV
    particle.pdg_encoding = I3Particle.ParticleType.unknown
    particle.pos = I3Position(*(reco_pars[d] * I3Units.m for d in ("x", "y", "z")))
    particle.shape = shape_map[particle_type]
    particle.time = reco_pars["time"] * I3Units.ns
    particle.speed = I3Constants.c

    if particle_type == "track":
        particle.length = particle.energy * TRACK_M_PER_GEV * I3Units.m
    else:
        particle.length = np.nan

    return particle
