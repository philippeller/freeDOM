"""Start or stop a service to use with the client module"""

import sys
from argparse import ArgumentParser

from freedom.reco.crs_reco import start_service
from freedom.llh_service.service_utils import kill_service

ctrl_addr = "tcp://127.0.0.1:*"
req_addr = "tcp://127.0.0.1:*"


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
    parser = ArgumentParser()
    parser.add_argument(
        "--hitnet",
        type=str,
        default="/cvmfs/icecube.opensciencegrid.org/users/peller/freeDOM/resources/HitNet_ranger_08_Dec_2021-10h53/epoch_50_model.hdf5",
        help="""hitnet file path""",
    )
    parser.add_argument(
        "--chargenet",
        type=str,
        default="/cvmfs/icecube.opensciencegrid.org/users/peller/freeDOM/resources/ChargeNet_ranger_23_Nov_2021-13h51/epoch_2000_model.hdf5",
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
