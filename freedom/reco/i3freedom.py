"""Provides I3Module(s) for FreeDOM reco"""

from freedom.reco.crs_reco import batch_crs_fit
from freedom.utils import i3frame_dataloader
from freedom.llh_service.llh_client import LLHClient
import numpy as np
import math
import time

DEFAULT_SEARCH_LIMITS = np.array(
    [
        [-500, 500],
        [-500, 500],
        [-1000, 700],
        [800, 20000],
        [0, 2 * math.pi],
        [0, math.pi],
        [0.1, 1000],
        [0, 1000],
    ]
).T
DEFAULT_INIT_RANGE = np.array(
    [
        [-50.0, 50.0],
        [-50.0, 50.0],
        [-100.0, 100.0],
        [-1000.0, 0.0],
        [0.0, 2 * math.pi],
        [0.0, math.pi],
        [0.0, 1.7],
        [0.0, 1.7],
    ]
)
DEFAULT_N_LIVE_POINTS = 97
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_ITER = 10000
DEFAULT_SPHERICAL_INDICES = [[4, 5]]


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
        print(f"Recoing this event took {delta*1000:.1f} ms")

        self.write_output_data(frame, suffix, fit_res)

    @staticmethod
    def write_output_data(frame, suffix, fit_res):
        """add reco output to an i3frame"""
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


def to_i3_vec(array, i3_vec_type):
    """convert a list/array to an I3Vec"""
    i3_vec = i3_vec_type()
    for val in array:
        i3_vec.append(val)
    return i3_vec
