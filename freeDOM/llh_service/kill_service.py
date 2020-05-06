"""
sends a "die" command to the llh service
causes a running llh service to exit its work loop
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import json
import sys

import zmq


def main():
    with open("service_params.json") as f:
        params = json.load(f)

    ctrl_sock = zmq.Context.instance().socket(zmq.PUSH)
    ctrl_sock.connect(params["ctrl_addr"])
    ctrl_sock.setsockopt(zmq.LINGER, 1000)

    ctrl_sock.send_string("die")


if __name__ == "__main__":
    sys.exit(main())
