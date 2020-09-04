"""
functions used for FreeDOM reconstruction

Provides a main() function to drive a reconstruction job
"""

__author__ = "Aaron Fienberg"

import argparse
import pickle
import math
import time
import json
import datetime
import pkg_resources
import functools
import sys


from multiprocessing import Process, Pool

import numpy as np

import pandas as pd

import zmq

from spherical_opt import spherical_opt

from freedom.llh_service.llh_client import LLHClient
from freedom.llh_service.llh_service import LLHService
from freedom.utils import i3cols_dataloader

NAN_REPLACE_VAL = 1e10


def get_out_of_bounds_func(limits):
    """returns func returning a boolean array, True for param rows that are out of bounds"""

    def out_of_bounds(params):
        return ~np.alltrue(
            np.logical_and(limits[0] <= params, params <= limits[1]), axis=-1
        )

    return out_of_bounds


def nan_replace(nll):
    # replace nans with valid, large values
    nll[np.isnan(nll)] = NAN_REPLACE_VAL
    return nll


def get_batch_closure(client, event, out_of_bounds):
    hit_data = event["hit_data"]
    evt_data = event["evt_data"]

    def eval_llh(params):
        llhs = client.eval_llh(hit_data, evt_data, params)

        llhs = np.atleast_1d(llhs)

        llhs = nan_replace(llhs)

        llhs[out_of_bounds(params)] = NAN_REPLACE_VAL

        return llhs

    return eval_llh


def initial_box(hits, init_range, charge_ind=4, n_params=8):
    """ returns initial box limits for each dimension
    in the form of a n_params x 2 table
    """

    # charge weighted positions, time
    hit_avgs = np.average(hits, weights=hits[:, charge_ind], axis=0)[:4]

    limits = np.empty((n_params, 2), np.float32)

    # x, y, z, t range from average + init_range[0] to average + init_range[1]
    limits[:4] = hit_avgs[:4, np.newaxis] + init_range[:4]

    # angles and energies just span the specified ranges
    # (although the energy parameters are log energies)
    limits[4:] = init_range[4:]

    return limits


def batch_crs_fit(
    event, client, rng, init_range, out_of_bounds, n_live_points, **sph_opt_kwargs
):

    eval_llh = get_batch_closure(client, event, out_of_bounds)

    n_params = len(init_range)

    box_limits = initial_box(event["hit_data"], init_range, n_params=n_params)

    uniforms = rng.uniform(size=(n_live_points, n_params))

    initial_points = box_limits[:, 0] + uniforms * (box_limits[:, 1] - box_limits[:, 0])

    # energy parameters need to be converted from log energy to energy
    initial_points[:, 6:] = 10 ** initial_points[:, 6:]

    opt_ret = spherical_opt.spherical_opt(
        func=eval_llh,
        method="CRS2",
        initial_points=initial_points,
        rand=rng,
        **sph_opt_kwargs,
    )

    return opt_ret


def fit_events(
    events,
    index,
    ctrl_addrs,
    init_range,
    search_limits,
    n_live_points,
    conf_timeout=60000,
    **sph_opt_kwargs,
):
    rng = np.random.default_rng()

    outputs = []

    client = LLHClient(ctrl_addr=ctrl_addrs[index], conf_timeout=conf_timeout)

    out_of_bounds = get_out_of_bounds_func(search_limits)

    for event in events:
        fit_res = batch_crs_fit(
            event,
            client,
            rng,
            init_range,
            out_of_bounds,
            n_live_points=n_live_points,
            **sph_opt_kwargs,
        )

        true_param_llh = client.eval_llh(
            event["hit_data"], event["evt_data"], event["params"]
        )
        retro_param_llh = client.eval_llh(
            event["hit_data"], event["evt_data"], event["retro"]
        )

        outputs.append((fit_res, true_param_llh, retro_param_llh))

    return outputs


def fit_event(
    event,
    ctrl_addr,
    init_range,
    search_limits,
    n_live_points,
    conf_timeout=60000,
    **sph_opt_kwargs,
):
    """wrapper around fit_events to fit a single event"""
    return fit_events(
        [event],
        0,
        [ctrl_addr],
        init_range,
        search_limits,
        n_live_points,
        conf_timeout,
        **sph_opt_kwargs,
    )[0]


def start_service(params, ctrl_addr, req_addr, gpu):
    # use a single GPU
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    params = params.copy()
    params["ctrl_addr"] = ctrl_addr
    params["req_addr"] = req_addr

    with LLHService(**params) as serv:
        print(f"starting service work loop for gpu {gpu}...")
        serv.start_work_loop()


def adjust_addr_string(base_str, gpu_ind):
    if base_str.startswith("ipc"):
        return f"{base_str}_{gpu_ind}"
    elif base_str.startswith("tcp"):
        split = base_str.split(":")
        port_num = int(split[-1])
        return f'{":".join(split[:-1])}:{port_num + int(gpu_ind)}'
    raise RuntimeError("only tcp and ipc addresses are supported")


def build_summary_df(all_outs, par_names):
    n_params = len(par_names)

    evt_idx = []
    free_fit_llhs = []
    true_param_llhs = []
    retro_param_llhs = []
    n_calls = []
    n_iters = []
    best_fit_ps = [[] for _ in range(n_params)]

    for i, out in enumerate(all_outs):
        freedom_params = out[0]["x"]
        freedom_llh = out[0]["fun"]
        n_calls.append(out[0]["n_calls"])
        n_iters.append(out[0]["nit"])

        evt_idx.append(i)
        free_fit_llhs.append(freedom_llh)
        true_param_llhs.append(out[1])
        retro_param_llhs.append(out[2])

        for p_ind, p in enumerate(freedom_params):
            best_fit_ps[p_ind].append(p)

    df_dict = dict(
        evt_idx=evt_idx,
        free_fit_llh=free_fit_llhs,
        true_p_llh=true_param_llhs,
        retro_p_llh=retro_param_llhs,
        n_calls=n_calls,
        n_iters=n_iters,
    )

    for p_name, p_list in zip(par_names, best_fit_ps):
        df_dict[p_name] = p_list

    return pd.DataFrame(df_dict)


def main():
    parser = argparse.ArgumentParser(description="runs a FreeDOM reconstruction job")

    parser.add_argument(
        "-c", "--conf_file", type=str, help="reco configuration file", required=True
    )
    args = parser.parse_args()

    with open(args.conf_file) as f:
        conf = json.load(f)

    i3cols_dirname = None
    try:
        i3cols_dirname = conf["i3cols_dir"]
        print(f"Loading i3cols data from {i3cols_dirname}")

        include_doms = "domnet_file" in conf["service_conf"]
        events = i3cols_dataloader.load_events(
            i3cols_dirname,
            recos={"retro": "retro_crs_prefit__median__neutrino"},
            include_doms=include_doms,
        )[0]

    except KeyError:
        print("i3cols_dir not specified. Looking for a test events pkl file")
        with open(conf["test_events_file"], "rb") as f:
            events = pickle.load(f)

    allowed_DOMs = np.load(
        pkg_resources.resource_filename("freedom", "resources/allowed_DOMs.npy")
    )
    ndoms = len(allowed_DOMs)

    service_conf = conf["service_conf"]

    # add hit_data, evt_data keys based on the networks being used
    for event in events:
        event["hit_data"] = event["hits"][:, : service_conf["n_hit_features"]]
        if event["hit_data"].shape[1] < service_conf["n_hit_features"]:
            # some networks were trained expecting more features than
            # are loaded in the most recent data loader functions
            n_additional_cols = (
                service_conf["n_hit_features"] - event["hit_data"].shape[1]
            )
            n_hits = len(event["hit_data"])
            event["hit_data"] = np.concatenate(
                (event["hit_data"], np.zeros((n_hits, n_additional_cols))), axis=1
            )

        if "domnet_file" in service_conf:
            event["evt_data"] = event["doms"][allowed_DOMs]
        else:
            event["evt_data"] = event["total_charge"]

    req_addrs = []
    ctrl_addrs = []
    gpus = conf["cuda_devices"]
    n_gpus = len(gpus)
    for gpu in gpus:
        req_addrs.append(adjust_addr_string(conf["base_req_addr"], gpu))
        ctrl_addrs.append(adjust_addr_string(conf["base_ctrl_addr"], gpu))

    print("starting LLH services...")
    procs = []
    for ctrl_addr, req_addr, gpu in zip(ctrl_addrs, req_addrs, gpus):
        proc = Process(
            target=start_service, args=(service_conf, ctrl_addr, req_addr, gpu)
        )
        proc.start()
        procs.append(proc)

    # wait for the LLH services to start by attempting to connect to them
    for ctrl_addr in ctrl_addrs:
        LLHClient(ctrl_addr=ctrl_addr, conf_timeout=60000)

    print("Services ready")

    # start the reco jobs
    pool_size = conf["n_workers"]
    evts_to_process = conf.get("n_evts", len(events))

    print(
        f"\nReconstructing {evts_to_process} events with {pool_size} workers and {n_gpus} gpus. Starting the jobs...\n"
    )

    evts_per_proc = int(math.ceil(evts_to_process / pool_size))
    evt_splits = [
        events[i * evts_per_proc : (i + 1) * evts_per_proc] for i in range(pool_size)
    ]

    worker_gpu_inds = np.arange(pool_size) % n_gpus

    init_range = np.array(conf["init_range"])
    param_search_limits = np.array(conf["param_search_limits"]).T
    n_live_points = conf["n_live_points"]
    conf_timeout = conf["conf_timeout"]
    sph_opt_kwargs = conf["spherical_opt_conf"]

    # fit events partial that fixes common parameters
    fit_events_partial = functools.partial(
        fit_events,
        ctrl_addrs=ctrl_addrs,
        init_range=init_range,
        search_limits=param_search_limits,
        n_live_points=n_live_points,
        conf_timeout=conf_timeout,
        **sph_opt_kwargs,
    )

    start = time.time()
    with Pool(pool_size) as p:
        outs = p.starmap(fit_events_partial, zip(evt_splits, worker_gpu_inds),)
    delta = time.time() - start
    print(f"reconstructing {evts_to_process} events took: {delta/60:.1f} minutes")

    # print summary results, save output file
    all_outs = sum((out for out in outs), [])

    print("Timing summary:")
    total_calls = sum(out[0]["n_calls"] for out in all_outs)
    total_iters = sum(out[0]["nit"] for out in all_outs)
    print(f"{total_calls} total calls")
    time_per_call = delta / total_calls
    print(f"{total_iters} total iters")
    time_per_iter = delta / total_iters
    print(f"{total_calls/len(all_outs):.1f} calls per event")
    print(f"{time_per_call*1e6:.2f} us per call")
    print(f"{total_iters/len(all_outs):.1f} iters per event")
    print(f"{time_per_iter*1e6:.2f} us per iter")

    print("\nSaving summary dataframe\n")
    # build summary df
    summary_df = build_summary_df(all_outs, conf["par_names"])
    # store some metadata
    summary_df.attrs["reco_conf"] = conf
    summary_df.attrs["reco_time"] = delta
    if i3cols_dirname is not None:
        summary_df.attrs["i3cols_dirname"] = i3cols_dirname

    # append datetime to the filename to avoid accidentally overwriting previous reco job's output
    time_str = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    outf_name = conf.get("outfile_name", "reco_out")
    outf_name = f"{outf_name}_{time_str}.pkl"
    summary_df.to_pickle(outf_name)

    print("Killing the LLH services")

    for proc, ctrl_addr in zip(procs, ctrl_addrs):
        with zmq.Context.instance().socket(zmq.REQ) as ctrl_sock:
            ctrl_sock.connect(ctrl_addr)
            ctrl_sock.send_string("die")
            proc.join()

    print("Done")


if __name__ == "__main__":
    sys.exit(main())
