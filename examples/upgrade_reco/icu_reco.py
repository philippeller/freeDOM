#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT combo/stable

"""Reconstruct ICU events with the I3FreedomClient"""

import pkg_resources
import sys
from argparse import ArgumentParser

from icecube import dataclasses, icetray, dataio
from I3Tray import I3Tray

import numpy as np
import zmq

from freedom.reco import i3freedom, transforms

CONF_TIMEOUT_MS = 10000

# filter out frames with fewer than 8 hits
def n_hits_filter(frame, reco_pulse_series_names):
    total_hits = 0
    for series_name in reco_pulse_series_names:
        pulses = frame[series_name]
        try:
            pulses = frame.apply(pulses)
        except AttributeError:
            pass

        hits_this_series = 0
        for pmt_pulses in pulses.values():
            hits_this_series += len(pmt_pulses)

        print(f"{series_name}: {hits_this_series} hits")

        total_hits += hits_this_series

    print(f"Found {total_hits} total hits")

    if total_hits >= 8:
        return True
    else:
        return False


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
        "--icecube_addr",
        type=str,
        required=True,
        help="""IceCube LLHService ctrl addr""",
    )
    parser.add_argument(
        "--mdom_addr", type=str, required=True, help="""mDOM LLHService ctrl addr"""
    )
    parser.add_argument(
        "--degg_addr", type=str, required=True, help="""DEgg LLHService ctrl addr"""
    )
    parser.add_argument(
        "--n_frames", type=int, default=None, help="""number of frames to process"""
    )
    parser.add_argument(
        "--gcd_file", type=str, default=None, help="""GCD file""",
    )

    args = parser.parse_args()

    geo = np.load(f"{args.resource_dir}/geo_array.npy")
    ug_geo = np.load(f"{args.resource_dir}/geo_array_upgrade.npy")
    mdom_directions = np.load(f"{args.resource_dir}/mdom_directions.npy")

    service_addrs = [args.icecube_addr, args.mdom_addr, args.degg_addr]
    pulse_series_names = ["IceCubePulsesTWSRT", "mDOMPulsesTWSRT", "DEggPulsesTWSRT"]

    try:
        freedom_reco = i3freedom.I3FreeDOMClient(service_addrs, CONF_TIMEOUT_MS)
    except zmq.error.Again:
        print("Could not connect to the LLH Service!")
        sys.exit(0)

    if args.gcd_file == None:
        files = args.input_files
    else:
        files = [args.gcd_file, args.input_files[0]]

    tray = I3Tray()
    tray.AddModule("I3Reader", FilenameList=files)
    tray.AddModule(n_hits_filter, reco_pulse_series_names=pulse_series_names)
    tray.AddModule(
        freedom_reco,
        geo=geo,
        reco_pulse_series_names=pulse_series_names,
        ug_geo=ug_geo,
        mdom_directions=mdom_directions,
        par_transforms=transforms.track_frac_transforms,
        do_track_dllh=True,
        suffix="ICU_test",
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
