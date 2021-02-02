"""
Fitting functions for CRS2-based FreeDOM event reconstruction 

Also provides a main() function to drive a CRS2-based reconstruction job 
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
import os

from multiprocessing import Process, Pool

import numpy as np
from scipy import constants
import tensorflow as tf

import zmq

from spherical_opt import spherical_opt

from freedom.llh_service.llh_client import LLHClient
from freedom.llh_service.llh_service import LLHService
from freedom.utils import i3cols_dataloader

from freedom.reco import bounds
from freedom.reco import summary_df
from freedom.reco import prefit

#load time residual spline function
from scipy.interpolate import UnivariateSpline
from freedom.utils.time_residual import TimeResidual
with open(pkg_resources.resource_filename('freedom', 'resources/prior/oscNext_time_residual_prior.pkl'), 'rb') as F:
    ftr = pickle.load(F)
ftr = UnivariateSpline._from_tck(ftr)
ftr.ext = 1


def get_batch_closure(clients, event, out_of_bounds, Tprior=True):
    """returns LLH batch eval closure for this event. The closure is a function of one argument: the hypotheses to evaluate"""
    hit_data = event["hit_data"]
    evt_data = event["evt_data"]
    
    def eval_llh(params):
        llhs = 0
        for i in range(len(clients)):
            llhs += np.atleast_1d(clients[i].eval_llh(hit_data[i], evt_data[i], params))
            
            if Tprior:
                tresiduals = TimeResidual(np.tile(hit_data[i], (len(llhs),1)).astype(np.float32), 
                                          np.repeat(params, len(hit_data[i]), axis=0).astype(np.float32))
                tresiduals = np.log(np.clip(ftr(tresiduals), 1e-10, 1))
                llhs -= np.sum(tresiduals.reshape((len(llhs), len(hit_data[i]))), axis=1)

        return bounds.invalid_replace(llhs, params, out_of_bounds)
    
    return eval_llh


def batch_crs_fit(
    event,
    clients,
    rng,
    init_range,
    search_limits,
    n_live_points,
    bounds_check_type="cube",
    Tprior=True,
    **sph_opt_kwargs,
):

    out_of_bounds = bounds.get_out_of_bounds_func(search_limits, bounds_check_type)

    eval_llh = get_batch_closure(clients, event, out_of_bounds, Tprior=Tprior)

    n_params = len(init_range)

    a = np.array([])
    for e in event['hit_data']:
        if len(e) == 0:
            continue
        if len(a) == 0:
            a = e
        else:
            a = np.append(a, e, axis=0)
    if len(a) == 0:
        a = np.array([[0, 0, -500, 9500, 1]])
    box_limits = prefit.initial_box(a, init_range, n_params=n_params)

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
    random_seed=None,
    conf_timeout=60000,
    Tprior=True,
    **sph_opt_kwargs,
):
    rng = np.random.default_rng(random_seed)

    outputs = []

    clients = [LLHClient(ctrl_addr=ctrl_addrs[index], conf_timeout=conf_timeout)]
    #clients = [] # use this for ICU reco
    #for i in range(3):
    #    clients.append(LLHClient(ctrl_addr=ctrl_addrs[i], conf_timeout=conf_timeout))

    for event in events:
        start = time.time()
        fit_res = batch_crs_fit(
            event,
            clients,
            rng,
            init_range,
            search_limits,
            n_live_points=n_live_points,
            Tprior=Tprior,
            **sph_opt_kwargs,
        )
        delta = time.time() - start
        
        true_param_llh = 0
        for i in range(len(clients)):
            true_param_llh += clients[i].eval_llh(
                event["hit_data"][i], event["evt_data"][i], event["params"]
            )
            if Tprior:
                tresidual = TimeResidual(event["hit_data"][i], np.array([event["params"]]))
                tresidual = np.log(np.clip(ftr(tresidual), 1e-10, 1))
                true_param_llh -= np.sum(tresidual)
        
        if "retro" in event.keys():
            retro_param_llh = clients[0].eval_llh(
                event["hit_data"], event["evt_data"], event["retro"]
            )
            if Tprior:
                tresidual = TimeResidual(event["hit_data"][i], np.array([event["retro"]]))
                tresidual = np.log(np.clip(ftr(tresidual), 1e-10, 1))
                retro_param_llh -= np.sum(tresidual)
            
            outputs.append((fit_res, true_param_llh, delta, retro_param_llh))
        else:
            outputs.append((fit_res, true_param_llh, delta))

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
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
    df = summary_df.build_summary_df(all_outs, conf["par_names"])
    # store some metadata
    df.attrs["reco_conf"] = conf
    df.attrs["reco_time"] = delta
    if i3cols_dirname is not None:
        df.attrs["i3cols_dirname"] = i3cols_dirname

    # append datetime to the filename to avoid accidentally overwriting previous reco job's output
    time_str = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    outf_name = conf.get("outfile_name", "reco_out")
    outf_name = f"{outf_name}_{time_str}.pkl"
    df.to_pickle(outf_name)

    print("Killing the LLH services")

    for proc, ctrl_addr in zip(procs, ctrl_addrs):
        with zmq.Context.instance().socket(zmq.REQ) as ctrl_sock:
            ctrl_sock.connect(ctrl_addr)
            ctrl_sock.send_string("die")
            proc.join()

    print("Done")


if __name__ == "__main__":
    sys.exit(main())
