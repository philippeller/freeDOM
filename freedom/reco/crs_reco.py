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

from multiprocessing import Process, Pool

import numpy as np

import zmq

from spherical_opt import spherical_opt

from freedom.llh_service.llh_client import LLHClient
from freedom.llh_service.service_utils import start_service
from freedom.utils import i3cols_dataloader

from freedom.reco import bounds
from freedom.reco import summary_df
from freedom.reco import prefit, postfit
from freedom.reco import transforms

DEFAULT_INIT_RANGE = np.array(
    [
        [-50.0, 50.0],
        [-50.0, 50.0],
        [-100.0, 100.0],
        [-1000.0, 0.0],
        [0.0, 2 * math.pi],
        [-1, 1],
        [0.0, 1.7],
        [0.0, 1.7],
    ]
)
DEFAULT_SEARCH_LIMITS = np.array(
    [
        [-500, 500],
        [-500, 500],
        [-1000, 700],
        [800, 20000],
        [0, 2 * math.pi],
        [0, math.pi],
        [0.1, 1000],
        [0, 1000],
    ]
).T

ZERO_TRACK_SEED_RANGE = 3
"""Range (in n estimated sigma) to smear best fit parameters before the zero track fit"""

TRACK_PAR_NAMES = ("track_energy", "track_frac")
"""known track energy related parameter names"""

CASC_PAR_NAMES = ("cascade_energy", "total_energy")
"""known total/cascade energy related parameter names"""

LOG_E_SEARCH_RANGE = (0, 2)
""" range of log energies over which to populate initial points in the zero track, energy only fit"""



def get_batch_closure(
    clients, event, out_of_bounds, param_transform=None, fixed_params=None
):
    """returns LLH batch eval closure for this event. 

    The closure is a function of one argument: the hypotheses to evaluate

    Parameters
    ----------
    clients : list of LLHClients
    event : dict
    out_of_bounds : callable
    param_transform : dict
    fixed_params : list of tuples
        tuples are of form (par_index, val_to_fix)
    """
    hit_data = event["hit_data"]
    evt_data = event["evt_data"]

    def eval_llh(params):
        trans_params = transforms.apply_transform(param_transform, params, fixed_params)

        llhs = 0
        for i in range(len(clients)):
            llhs += np.atleast_1d(
                clients[i].eval_llh(hit_data[i], evt_data[i], trans_params)
            )

        llhs = bounds.invalid_replace(llhs, trans_params, out_of_bounds)

        for param, llh in zip(params, llhs):
            eval_llh.evaluated_pts.append(np.hstack((param, [llh])))

        return llhs

    eval_llh.evaluated_pts = []

    return eval_llh


def batch_crs_fit(
    event,
    clients,
    rng,
    init_range=DEFAULT_INIT_RANGE,
    search_limits=DEFAULT_SEARCH_LIMITS,
    n_live_points=None,
    bounds_check_type="cube",
    do_postfit=False,
    store_all=False,
    truth_seed=False,
    seed=None,
    param_transforms=None,
    fixed_params=None,
    initial_points=None,
    **sph_opt_kwargs,
):
    """fit an event with the CRS2 optimizer
    
    Parameters
    ----------
    event : dict
        the event data
    clients : list
        list of LLHClients, one per DOM type
    rng : np.random.Generator           
    init_range : np.ndarray
    search_limits : np.ndarray
    n_live_poins : int, default None
        must be specified if initial_points is not
    bounds_check_type : str, default "cube"
    do_postfit : bool
    store_all : bool
        whether to include all sampled LLH points in the return dct
    truth_seed : bool
    seed: np.ndarray, default None
    param_transforms : dict, default None
    fixed_params : list, default None
        list of form [(par_index, fix_value)]
    initial_points : np.ndarray, default None
        supersedes init_range, truth_seed, n_live_points
    **sph_opt_kwargs
        kwargs that will be passed to spherical_opt.spherical_opt()

    Returns
    -------
    dict
        fit result
    """

    out_of_bounds = bounds.get_out_of_bounds_func(search_limits, bounds_check_type)

    if param_transforms is None:
        trans = None
        inv_trans = None
    else:
        trans = param_transforms["trans"]
        inv_trans = param_transforms["inv_trans"]

    eval_llh = get_batch_closure(clients, event, out_of_bounds, trans, fixed_params)

    n_params = len(init_range)

    # for ICU reco it can happend that we have empty hit data
    all_hits = np.array([])
    for hits in event["hit_data"]:
        if len(hits) == 0:
            continue
        if len(all_hits) == 0:
            all_hits = hits
        else:
            all_hits = np.append(all_hits, hits, axis=0)
    if len(all_hits) == 0:
        all_hits = np.array([[0, 0, -500, 9500, 1]])

    if initial_points is None:
        if n_live_points is None:
            raise ValueError(
                "'n_live_points' must be specified when 'initial_points' is not set!"
            )

        if truth_seed:
            box_limits = prefit.truth_seed_box(event["params"], init_range)
        elif np.all(seed) != None:
            box_limits = prefit.truth_seed_box(seed, init_range)
        else:
            box_limits = prefit.initial_box(all_hits, init_range, n_params=n_params)

        uniforms = rng.uniform(size=(n_live_points, n_params))

        initial_points = box_limits[:, 0] + uniforms * (
            box_limits[:, 1] - box_limits[:, 0]
        )

        # for non truth seed, convert from cos zenith to zenith
        if not truth_seed and np.all(seed) == None:
            initial_points[:, 5] = np.arccos(initial_points[:, 5])
        
        # energy parameters need to be converted from log energy to energy
        initial_points[:, 6:] = 10 ** initial_points[:, 6:]

        if truth_seed:
            initial_points[-1] = event["params"]
        elif np.all(seed) != None:
            initial_points[-1] = seed

        if inv_trans is not None:
            initial_points = inv_trans(initial_points)

    opt_ret = spherical_opt.spherical_opt(
        func=eval_llh,
        method="CRS2",
        initial_points=initial_points,
        rand=rng,
        **sph_opt_kwargs,
    )

    if store_all:
        opt_ret["all_pts"] = eval_llh.evaluated_pts

    if do_postfit:
        opt_ret["postfit"] = postfit.postfit(
            eval_llh.evaluated_pts,
            par_names=transforms.free_par_names(param_transforms, fixed_params),
        )

    return opt_ret


def fit_events(
    events,
    index,
    ctrl_addrs,
    init_range=DEFAULT_INIT_RANGE,
    search_limits=DEFAULT_SEARCH_LIMITS,
    n_live_points=None,
    random_seed=None,
    conf_timeout=60000,
    do_postfit=False,
    store_all=False,
    truth_seed=False,
    seeds=None,
    param_transforms=None,
    fixed_params=None,
    initial_points=None,
    **sph_opt_kwargs,
):
    """fit a list of events
    
    see batch_opt_ret for param descriptions"""

    rng = np.random.default_rng(random_seed)

    outputs = []

    if isinstance(ctrl_addrs[index], str):
        clients = [LLHClient(ctrl_addr=ctrl_addrs[index], conf_timeout=conf_timeout)]
    else:
        clients = [LLHClient(ctrl_addr=addr, conf_timeout=conf_timeout) for addr in ctrl_addrs[index]]
    
    if np.all(seeds) == None:
        seeds = [None] * len(events)

    for j, event in enumerate(events):
        fit_res = timed_fit(
            event,
            clients,
            rng,
            init_range,
            search_limits,
            n_live_points=n_live_points,
            do_postfit=do_postfit,
            store_all=store_all,
            truth_seed=truth_seed,
            seed=seeds[j],
            param_transforms=param_transforms,
            fixed_params=fixed_params,
            initial_points=initial_points,
            **sph_opt_kwargs,
        )
        delta = fit_res["delta"]

        try:
            true_param_llh = 0
            for i in range(len(clients)):
                true_param_llh += clients[i].eval_llh(
                    event["hit_data"][i], event["evt_data"][i], event["params"]
                )
        except KeyError:
            # true params not available
            true_param_llh = None

        if "retro" in event.keys():
            retro_param_llh = clients[0].eval_llh(
                event["hit_data"], event["evt_data"], event["retro"]
            )

            outputs.append((fit_res, true_param_llh, delta, retro_param_llh))
        else:
            outputs.append((fit_res, true_param_llh, delta))

    return outputs


def fit_event(
    event,
    ctrl_addr,
    init_range=DEFAULT_INIT_RANGE,
    search_limits=DEFAULT_SEARCH_LIMITS,
    n_live_points=None,
    conf_timeout=60000,
    do_postfit=False,
    store_all=False,
    truth_seed=False,
    param_transforms=None,
    fixed_params=None,
    initial_points=None,
    **sph_opt_kwargs,
):
    """wrapper around fit_events to fit a single event"""
    return fit_events(
        [event],
        index=0,
        ctrl_addrs=[ctrl_addr],
        init_range=init_range,
        search_limits=search_limits,
        n_live_points=n_live_points,
        conf_timeout=conf_timeout,
        do_postfit=do_postfit,
        store_all=store_all,
        truth_seed=truth_seed,
        param_transforms=param_transforms,
        fixed_params=fixed_params,
        initial_points=initial_points,
        **sph_opt_kwargs,
    )[0]


def timed_fit(*fit_args, **fit_kwargs):
    """wrapper around batch_crs_fit that records the fit's wall clock duration"""
    start = time.time()
    fit_res = batch_crs_fit(*fit_args, **fit_kwargs)
    fit_res["delta"] = time.time() - start

    return fit_res


def zero_track_fit(full_res, *fit_args, **fit_kwargs):
    """conduct no-track fits based on the result of a full fit
    
    Parameters
    ----------
    full_res : dict
        full crs fit result; must include postfit
    *fit_args 
        positional args for batch_crs_fit
    **fit_kwargs 
        arbitrary kwargs for batch_crs_fit

    Returns
    -------
    tuple
        (no track fit result, no track energy only fit result)
    """
    par_transforms = fit_kwargs.get("param_transforms", None)
    rng = fit_kwargs.get("rng", None)
    n_live_points = fit_kwargs["n_live_points"]
    batch_size = fit_kwargs["batch_size"]

    par_names = transforms.free_par_names(par_transforms, None)

    # zero track fit with all other params free
    fixed_pars = [
        (i, 0.0) for i, name in enumerate(par_names) if name in TRACK_PAR_NAMES
    ]
    if len(fixed_pars) != 1:
        raise ValueError(
            f"Expected exactly one track-E like par name, found {len(fixed_pars)}"
        )
    fix_ind = fixed_pars[0][0]
    # this routine will break if track par index is < the spherical indices
    # for now, let's just check that isn't the case
    all_sph_inds = sum(fit_kwargs.get("spherical_indices", [[]]), [])
    if any(fix_ind < sph_ind for sph_ind in all_sph_inds):
        raise ValueError(
            "zero_track_fit requires spherical par indices to be < energy par indices"
        )

    best_fit = full_res["x"]
    seed_pars = np.array([x for i, x in enumerate(best_fit) if i != fix_ind])

    stds = full_res["postfit"]["stds"]
    seed_stds = np.array([std for i, std in enumerate(stds) if i != fix_ind])

    initial_points = prefit.seed_box(
        seed_pars, ZERO_TRACK_SEED_RANGE * seed_stds, n_live_points, rng
    )

    fit_kwargs.update(dict(fixed_params=fixed_pars, initial_points=initial_points))

    no_track_res = timed_fit(*fit_args, **fit_kwargs)
    no_track_res["fixed_params"] = fixed_pars.copy()

    # zero track fit with only total / cascade energy free
    for i, name in enumerate(par_names):
        if name not in CASC_PAR_NAMES and name not in TRACK_PAR_NAMES:
            fixed_pars.append((i, best_fit[i]))

    n_points = batch_size + 1
    initial_points = np.logspace(*LOG_E_SEARCH_RANGE, n_points)[:, np.newaxis]

    fit_kwargs.update(
        dict(
            fixed_params=fixed_pars, initial_points=initial_points, spherical_indices=()
        )
    )

    E_only_res = timed_fit(*fit_args, **fit_kwargs)
    E_only_res["fixed_params"] = fixed_pars

    return no_track_res, E_only_res


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

        # adapt to structure created for ICU reco
        event["hit_data"] = [event["hit_data"]]

        if "domnet_file" in service_conf:
            event["evt_data"] = [event["doms"][allowed_DOMs]]
        else:
            event["evt_data"] = [event["total_charge"]]

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
    do_postfit = conf["do_postfit"]
    truth_seed = conf.get("truth_seed", False)

    # fit events partial that fixes common parameters
    fit_events_partial = functools.partial(
        fit_events,
        ctrl_addrs=ctrl_addrs,
        init_range=init_range,
        search_limits=param_search_limits,
        n_live_points=n_live_points,
        conf_timeout=conf_timeout,
        do_postfit=do_postfit,
        truth_seed=truth_seed,
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
