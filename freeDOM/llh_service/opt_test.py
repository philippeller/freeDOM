#!/usr/bin/env python

"""
Fake optimizer (simple grid search) for testing the llh_service.
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import argparse
import json
import os
import sys
import time
import math

import numpy as np

from llh_client import LLHClient


GRIDSIZE = 60


def opt_test(client, mean, std, n_obs, serial, gridsize=GRIDSIZE, plot=True):
    """Run a simple "optimization" (grid search) that requests a lot of LLH's
    from the llh_service.

    Parameters
    ----------
    client : freeDOM.llh_service.llh_client.LLHClient
    mean : float
        True-Gaussian's mean
    std : float
        True-Gaussian's standard deviation
    n_obs : int
        Number of observations (float32 values) per hypothesis
    serial : bool
        Request LLH's one at a time from the server
    gridsize : int, optional
        gridsize x gridsize points are sampled in (mu, sigma)-space. Default is
        60, or 3600 total points, similar to the number of LLH evals per event
        in Retro
    plot : bool, optional
        Plot the results

    """

    # generate the random samples
    rand = np.random.default_rng(seed=0)
    x = rand.normal(loc=mean, scale=std, size=n_obs).astype(np.float32)

    # do a grid search over mu, sigma around the sample mean, std
    samp_mean = np.mean(x)
    samp_std = np.std(x)

    err_mean = samp_std / math.sqrt(len(x))

    means = np.linspace(
        samp_mean - 4.5 * err_mean, samp_mean + 4.5 * err_mean, gridsize
    )
    stds = np.linspace(samp_std * 0.85, samp_std * 1.15, gridsize)

    llh_map = np.empty((len(stds), len(means)), dtype=np.float32)

    start = 0
    if serial:
        # fill the map using synchronous llh evaluations
        start = time.time()
        for i, std_ in enumerate(stds):
            for j, mean_ in enumerate(means):
                llh_map[i, j] = client.eval_llh(x, mean_, std_)
    else:
        mg_stds, mg_means = np.meshgrid(stds, means)
        n_hypos = mg_means.size

        batch_size = client.max_hypos_per_batch
        num_requests = 0

        start = time.time()
        for start_idx in range(0, n_hypos, batch_size):
            client.request_batch_eval(
                x,
                mg_means.flat[start_idx : start_idx + batch_size],
                mg_stds.flat[start_idx : start_idx + batch_size],
                req_id=str(start_idx),
            )
            num_requests += 1

        num_replies = 0
        while num_replies < num_requests:
            reply = client.recv()
            if reply is not None:
                start_idx = int(reply["req_id"])
                llh_map.flat[start_idx : start_idx + batch_size] = reply["llh"]
                num_replies += 1

    delta = time.time() - start

    n_eval = len(stds) * len(means)

    print(
        f"{n_eval} evals took {delta*1000:.3f} ms"
        f" ({delta/n_eval*1e3:.3f} ms per eval)"
    )

    if plot:
        import matplotlib as mpl

        mpl.use("agg")
        from matplotlib import pyplot as plt

        os.makedirs("plots", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pcolormesh(means, stds, llh_map - llh_map.max(), vmin=-10, vmax=0)
        ax.contour(
            means,
            stds,
            llh_map.max() - llh_map,
            [2.28 / 2, 4.61 / 2, 5.99 / 2],
            colors=["blue", "blue", "blue"],
            linestyles=["solid", "dashed", "dotted"],
        )

        ax.set_xlabel(r"$\mu$", fontsize=16)
        ax.set_ylabel(r"$\sigma$", fontsize=16)
        ax.set_title(fr"$\mu$ = {mean:.1f}, $\sigma$ = {std:.1f}", fontsize=24)
        ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(f"plots/opt_test_{mean}_{std}_{n_obs}.png", bbox="tight")


def main(description=__doc__):
    """Script interface to `opt_test` function"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mean", type=float, help="True Gaussian distribution's mean")
    parser.add_argument(
        "std", type=float, help="True Gaussian distribution's standard deviation"
    )
    parser.add_argument(
        "n_obs", type=int, help="Number of observations (float32 values) per hypothesis"
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=GRIDSIZE,
        help="Evaluate gridsize x gridsize LLH's in (mu, sigma)-space",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Request LLH's one at a time (otherwise, all are requested at once)",
    )
    parser.add_argument("--plot", action="store_true", help="Whether to plot results")
    parser.add_argument(
        "--service-params-file",
        metavar="SERVICE_PARAMS_JSON",
        default="service_params.json",
        help="JSON file containing parameters for running service and clients",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    service_params_file = kwargs.pop("service_params_file")
    service_params_file = os.path.expanduser(os.path.expandvars(service_params_file))
    with open(service_params_file) as f:
        params = json.load(f)
    client = LLHClient(req_addr=params["req_addr"], batch_size=params["batch_size"])

    opt_test(client=client, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
