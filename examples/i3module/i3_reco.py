#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT combo/stable

"""Reconstruct events with the I3FreedomClient"""

import pkg_resources
import sys
from argparse import ArgumentParser

from icecube import dataclasses, icetray, dataio
from I3Tray import I3Tray

import numpy as np
import zmq

from freedom.reco import i3freedom, transforms

CONF_TIMEOUT_MS = 10000

# for counting events
def evt_counter(frame):
    evt_counter.evt_num += 1
    print(f"Finished event {evt_counter.evt_num}")


evt_counter.evt_num = 0


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
        default=pkg_resources.resource_filename("freedom", "resources"),
    )
    parser.add_argument(
        "--service_addr", type=str, required=True, help="""LLHService ctrl addr"""
    )
    parser.add_argument(
        "--n_frames", type=int, default=None, help="""number of frames to process"""
    )
    parser.add_argument(
        "--gcd_file", type=str, default=None, help="""GCD file""",
    )

    args = parser.parse_args()

    geo = np.load(f"{args.resource_dir}/geo_array.npy")

    try:
        freedom_reco = i3freedom.I3FreeDOMClient(args.service_addr, CONF_TIMEOUT_MS)
    except zmq.error.Again:
        print("Could not connect to the LLH Service!")
        sys.exit(0)
    
    if args.gcd_file == None:
        files = args.input_files
    else:
        files = [args.gcd_file, args.input_files[0]]
    
    tray = I3Tray()
    tray.AddModule("I3Reader", FilenameList=files)
    tray.AddModule(
        freedom_reco,
        geo=geo,
        reco_pulse_series_names="SRTTWOfflinePulsesDC",
        par_transforms=transforms.track_frac_transforms,
        do_track_dllh=True,
        suffix="test",
    )
    tray.AddModule(evt_counter)
    tray.AddModule(
        "I3Writer", DropOrphanStreams=[icetray.I3Frame.DAQ], filename=args.output_file
    )

    if args.n_frames is None:
        tray.Execute()
    else:
        tray.Execute(args.n_frames)


if __name__ == "__main__":
    main()
