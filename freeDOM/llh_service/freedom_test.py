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
import pickle

import numpy as np

from llh_client import LLHClient

N_ITERATIONS = 3600


def main():
    with open("service_params.json") as f:
        params = json.load(f)

    client = LLHClient(
        req_addr=params["req_addr"],
        batch_size=params["batch_size"],
        n_hypo_params=params["n_hypo_params"],
        n_obs_features=params["n_obs_features"],
    )

    # test a single synchronous eval

    with open("../test_data/test_event.pkl", "rb") as f:
        event = pickle.load(f)

    hits = event["hits"].flatten()
    theta = event["params"].flatten()

    llhs = []
    start = time.time()
    for i in range(N_ITERATIONS):
        llhs.append(client.eval_llh(hits, theta))
    delta = time.time() - start

    print(
        f"{N_ITERATIONS} evals took {delta*1000:.3f} ms"
        f" ({delta/N_ITERATIONS*1e3:.3f} ms per eval)"
    )


if __name__ == "__main__":
    sys.exit(main())
