"""Start or stop a service to use with the client module"""

import pkg_resources
import sys
from argparse import ArgumentParser

import zmq

from freedom.reco.crs_reco import start_service, DEFAULT_SEARCH_LIMITS
from freedom.neural_nets.transformations import prior_trafo


ctrl_addr = "tcp://127.0.0.1:*"
req_addr = "tcp://127.0.0.1:*"


def kill_service(ctrl_addr):
    with zmq.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(ctrl_addr)

        sock.send_string("die")

        try:
            print(f"service response: {sock.recv_string()}")
        except zmq.error.Again:
            print("No response from the LLH service")


def build_service_conf(hitnet, chargenet):
    return {
        "poll_timeout": 1,
        "flush_period": 1,
        "n_hypo_params": 8,
        "n_hit_features": 10,
        "n_evt_features": 2,
        "batch_size": {"n_hypos": 400, "n_observations": 12000},
        "send_hwm": 10000,
        "recv_hwm": 10000,
        "hitnet_file": hitnet,
        "chargenet_file": chargenet,
    }


def main():
    resource_dir = pkg_resources.resource_filename("freedom", "resources")

    parser = ArgumentParser()
    parser.add_argument(
        "--hitnet",
        type=str,
        default="/cvmfs/icecube.opensciencegrid.org/users/peller/freeDOM/resources/HitNet_ranger_06_Sep_2021-18h50/epoch_25_model.hdf5",
        help="""hitnet file path""",
    )
    parser.add_argument(
        "--chargenet",
        type=str,
        default="/cvmfs/icecube.opensciencegrid.org/users/peller/freeDOM/resources/ChargeNet_ranger_23_Aug_2021-18h42/epoch_1500_model.hdf5",
        help="""chargenet file path""",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="""cuda device index"""
    )
    parser.add_argument(
        "--ctrl_addr",
        type=str,
        default=None,
        help="""ctrl addr (for killing a service)""",
    )
    parser.add_argument(
        "--kill", action="store_true", help="""kill a running LLH service"""
    )

    args = parser.parse_args()

    service_conf = build_service_conf(args.hitnet, args.chargenet)

    if args.kill:
        if args.ctrl_addr is None:
            print("Killing a service requires a ctrl addr!")
            sys.exit(0)
        print(f"Killing service at {args.ctrl_addr}")
        kill_service(args.ctrl_addr)
    else:
        start_service(
            service_conf,
            ctrl_addr=ctrl_addr,
            req_addr=req_addr,
            cuda_device=args.cuda_device,
        )


if __name__ == "__main__":
    main()
