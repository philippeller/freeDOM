"""Start LLH services on a single GPU and then launch ICU reco processes"""

from argparse import ArgumentParser
import multiprocessing
import os
import subprocess
import sys

from freedom.llh_service.service_utils import start_service_pipe, kill_service

OBS_PER_HYPO = {"DOM": 16, "mDOM": 64, "DEgg": 32}


def build_ICU_service_conf(hitnet, chargenet, obs_per_hypo, n_hypos=400):
    return {
        "poll_timeout": 1,
        "flush_period": 1,
        "n_hypo_params": 8,
        "n_hit_features": 10,
        "n_evt_features": 2,
        "batch_size": {"n_hypos": n_hypos, "n_observations": n_hypos * obs_per_hypo},
        "send_hwm": 10000,
        "recv_hwm": 10000,
        "hitnet_file": hitnet,
        "chargenet_file": chargenet,
        "ctrl_addr": "tcp://127.0.0.1:*",
        "req_addr": "tcp://127.0.0.1:*",
    }


def start_services(model_dir, om_types, cuda_device):
    service_procs = []
    pipes = []
    for om_type in om_types:
        hitnet = f"{model_dir}/hitNet_{om_type}s.hdf5"
        chargenet = f"{model_dir}/chargeNet_{om_type}s.hdf5"
        conf = build_ICU_service_conf(hitnet, chargenet, OBS_PER_HYPO[om_type])

        read_end, write_end = multiprocessing.Pipe()
        proc = multiprocessing.Process(
            target=start_service_pipe, args=(conf, cuda_device, write_end)
        )
        proc.start()

        service_procs.append(proc)
        pipes.append(read_end)

    ctrl_addrs = [pipe.recv() for pipe in pipes]
    return service_procs, ctrl_addrs


def start_reco_process(infile, outdir, ctrl_addrs, n_frames=None):
    """start a single reco process"""
    filename = os.path.basename(infile)
    extension_ind = filename.rfind(".i3.zst")
    if extension_ind == -1:
        extension_ind = len(infile)
    outfile = f"{outdir}/{filename[:extension_ind]}_reco.i3.zst"

    cmd = f"./icu_reco.py --input_files {infile} --output_file {outfile}".split()
    cmd.extend(["--ctrl_addrs"] + ctrl_addrs)

    if n_frames is not None:
        cmd.extend(["--n_frames", f"{n_frames}"])

    return subprocess.Popen(cmd)


def start_reco_processes(inlist, outdir, ctrl_addrs, n_frames=None):
    """start a reco process for each file in `inlist"""
    with open(inlist, "r") as f:
        return [
            start_reco_process(fname.strip(), outdir, ctrl_addrs, n_frames)
            for fname in f
        ]


def main():
    cvmfs_dir = "/cvmfs/icecube.opensciencegrid.org/users/peller/freeDOM/resources"
    model_dir = f"{cvmfs_dir}/Upgrade_NNs"

    parser = ArgumentParser()
    parser.add_argument(
        "--in_list", type=str, required=True, help="""Input file list"""
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="""Output directory"""
    )
    parser.add_argument(
        "--cuda_device", type=int, required=True, help="""cuda device index"""
    )
    parser.add_argument(
        "--n_frames", type=int, default=None, help="""number of frames to process"""
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=model_dir,
        help="""Upgrade NN model path""",
    )
    parser.add_argument(
        "--om_types",
        type=str,
        nargs="+",
        default=["DOM", "mDOM", "DEgg"],
        help="""optical module types""",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        print(f"{args.outdir} is not a valid directory", file=sys.stderr)
        sys.exit(0)

    service_procs, ctrl_addrs = start_services(
        args.model_dir, args.om_types, args.cuda_device
    )

    try:
        reco_procs = start_reco_processes(
            args.in_list, args.outdir, ctrl_addrs, args.n_frames
        )
        for proc in reco_procs:
            proc.wait()
    finally:
        for p, addr in zip(service_procs, ctrl_addrs):
            kill_service(addr)
            p.join()


if __name__ == "__main__":
    main()
