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

from freedom.llh_service.llh_client import LLHClient

N_ITERATIONS = 3600


def main():
    with open("service_params.json") as f:
        params = json.load(f)

    client = LLHClient(ctrl_addr=params["ctrl_addr"], conf_timeout=20000)

    with open("../resources/test_data/test_events.pkl", "rb") as f:
        event = pickle.load(f)[8]

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
