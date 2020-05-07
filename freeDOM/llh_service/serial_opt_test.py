#!/usr/bin/env python

"""
Fake serial optimizer

For testing the llh_service
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import json
import os
import sys
import time
import math

from matplotlib import pyplot as plt
import numpy as np

from llh_client import LLHClient

# 60^2 = 3600, similar to the number of LLH evals per event in retro
GRIDSIZE = 60


def main():
    if len(sys.argv) < 4:
        print("Usage: serial_opt_test.py <mu> <sig> <n_obs>")
        sys.exit(0)

    with open("service_params.json") as f:
        params = json.load(f)

    client = LLHClient(req_addr=params["req_addr"], batch_size=params["batch_size"])

    # parameters of a Gaussian distribution
    mu = float(sys.argv[1])
    sig = float(sys.argv[2])

    # number of "observations", or samples,
    # in the test experiment
    n_obs = int(sys.argv[3])

    # generate the random samples
    x = np.empty(n_obs, np.float32)
    x[:] = sig * np.random.randn(n_obs) + mu

    # do a grid search over mu, sigma around the sample mean, std
    samp_mean = np.mean(x)
    samp_std = np.std(x)

    err_mean = samp_std / math.sqrt(len(x))

    mean_range = np.linspace(
        samp_mean - 4.5 * err_mean, samp_mean + 4.5 * err_mean, GRIDSIZE
    )
    std_range = np.linspace(samp_std * 0.85, samp_std * 1.15, GRIDSIZE)

    llh_map = np.empty((len(std_range), len(mean_range)), dtype=np.float32)

    start = time.time()

    # fill the map using synchronous llh evaluations
    for i, std in enumerate(std_range):
        for j, mean in enumerate(mean_range):
            llh_map[i, j] = client.eval_llh(x, mean, std)

    delta = time.time() - start

    n_eval = GRIDSIZE ** 2

    print(
        f"{n_eval} evals took {delta*1000:.3f} ms"
        f" ({delta/n_eval*1e3:.3f} ms per eval)"
    )

    try:
        os.mkdir("plots")
    except OSError as err:
        if err.errno != 17:
            raise

    fig = plt.figure(figsize=(6, 6))

    ax = plt.subplot(111)

    ax.pcolormesh(mean_range, std_range, llh_map - llh_map.max(), vmin=-10, vmax=0)
    ax.contour(
        mean_range,
        std_range,
        llh_map.max() - llh_map,
        [2.28 / 2, 4.61 / 2, 5.99 / 2],
        colors=["blue", "blue", "blue"],
        linestyles=["solid", "dashed", "dotted"],
    )

    ax.set_xlabel("$\mu$", fontsize=16)
    ax.set_ylabel("$\sigma$", fontsize=16)
    ax.set_title(f"$\mu$ = {mu:.1f}, $\sigma$ = {sig:.1f}", fontsize=24)
    ax.tick_params(labelsize=16)

    plt.savefig(f"plots/serial_test_{mu}_{sig}_{n_obs}.png", bbox="tight")


if __name__ == "__main__":
    sys.exit(main())
