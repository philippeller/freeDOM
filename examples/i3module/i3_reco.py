"""Reconstruct events with the I3FreedomClient"""

from argparse import ArgumentParser
from icecube import dataclasses, icetray, dataio
from I3Tray import I3Tray

import numpy as np
import zmq

import sys

# temporary solution
sys.path.append("/home/atfienberg/IceCube/freeDOM")
sys.path.append("/home/atfienberg/IceCube/spherical_opt")

from freedom.reco import i3freedom
from freedom.utils.i3frame_dataloader import load_event

CONF_TIMEOUT_MS = 10000


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input_files", type=str, required=True, nargs="+", help="""Input I3 file""",
    )
    parser.add_argument(
        "--output_file", type=str, default="test.i3.zst", help="""output file name""",
    )
    parser.add_argument(
        "--resource_dir",
        type=str,
        help="""Resource directory""",
        default="/home/atfienberg/IceCube/freeDOM/freedom/resources",
    )
    parser.add_argument(
        "--service_addr", type=str, required=True, help="""LLHService ctrl addr"""
    )

    args = parser.parse_args()

    geo = np.load(f"{args.resource_dir}/geo_array.npy")

    try:
        freedom_reco = i3freedom.I3FreeDOMClient(args.service_addr, CONF_TIMEOUT_MS)
    except zmq.error.Again:
        print("Could not connect to the LLH Service!")
        sys.exit(0)

    tray = I3Tray()
    tray.AddModule("I3Reader", FilenameList=args.input_files)
    tray.AddModule(
        freedom_reco,
        geo=geo,
        reco_pulse_series_name="SRTTWOfflinePulsesDC",
        suffix="test",
    )
    tray.AddModule(
        "I3Writer", DropOrphanStreams=[icetray.I3Frame.DAQ], filename=args.output_file
    )
    tray.Execute(50)


if __name__ == "__main__":
    main()
