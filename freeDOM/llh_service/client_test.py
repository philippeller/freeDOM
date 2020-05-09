#!/usr/bin/env python

"""
test of llh_client/server communication
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import json
import os
import sys
import time

import matplotlib as mpl

mpl.use("agg")
from matplotlib import pyplot as plt
import numpy as np

from llh_client import LLHClient


def main():
    if len(sys.argv) < 3:
        print("Usage: client_test.py <mu> <n_obs>")
        sys.exit(0)

    with open("service_params.json") as f:
        params = json.load(f)

    client = LLHClient(req_addr=params["req_addr"], batch_size=params["batch_size"])

    # parameters of a Gaussian distribution
    mu = float(sys.argv[1])
    sig = 1

    # number of "observations", or samples,
    # in the test experiment
    n_obs = int(sys.argv[2])

    # generate the random samples
    x = np.empty(n_obs, np.float32)
    x[:] = sig * np.random.randn(n_obs) + mu

    # test a single asynchronous evaluation
    now = time.time()
    client.request_eval(x, mu, sig, req_id="test_id")
    delta = time.time() - now
    print(f"request took {delta*1000:.3f} ms")

    reply = None
    while reply is None:
        reply = client.recv()

    print(reply)

    # test a single synchronous evaluation
    now = time.time()
    llh = client.eval_llh(x, mu, sig)
    delta = time.time() - now
    print(f"single LLH eval took {delta*1000:.3f} ms")
    print(llh)

    # test a lot of LLH evaluations
    n_eval = 40000

    # in batches of the max size
    batch_size = client.max_hypos_per_batch

    mus = np.linspace(-1.0, 1.0, n_eval).reshape(int(n_eval / batch_size), batch_size)
    sigs = np.repeat(sig, batch_size)

    now = time.time()
    for i, test_mus in enumerate(mus):
        client.request_batch_eval(x, test_mus, sigs, req_id=str(i))

    replies = []
    while len(replies) < len(mus):
        reply = client.recv()
        if reply is not None:
            replies.append(reply)

    llhs = np.hstack([r["llh"] for r in replies])
    delta = time.time() - now
    print(
        f"{n_eval} evals took {delta*1000:.3f} ms"
        f" ({delta/n_eval*1e6:.3f} us per eval)"
    )

    os.makedirs("plots", exist_ok=True)
    plt.plot(mus.flatten(), llhs)
    plt.savefig(f"plots/test_{mu}_{n_obs}.png", bbox="tight")


if __name__ == "__main__":
    sys.exit(main())
