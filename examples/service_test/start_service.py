#!/usr/bin/env python

"""
starts the LLH service
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import argparse
import json
import sys

from freedom.llh_service.llh_service import LLHService


def main():
    parser = argparse.ArgumentParser(description="Starts the LLH service.")
    parser.add_argument(
        "-c", "--conf_file", type=str, help="service configuration file", required=True
    )
    args = parser.parse_args()

    with open(args.conf_file) as f:
        params = json.load(f)

    with LLHService(**params) as service:
        print("starting work loop:\n")
        sys.stdout.flush()
        service.start_work_loop()


if __name__ == "__main__":
    sys.exit(main())
