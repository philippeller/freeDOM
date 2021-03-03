"""Start or stop a service to use with the client module"""

import sys
from argparse import ArgumentParser

import zmq

from freedom.reco.crs_reco import start_service, adjust_addr_string
from freedom.reco.i3freedom import DEFAULT_SEARCH_LIMITS

ctrl_addr = "tcp://127.0.0.1:9887"
req_addr = "tcp://127.0.0.1:10101"


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


def build_service_conf(hitnet, chargenet, theta_prior, t_prior):
    return {
        "poll_timeout": 1,
        "flush_period": 1,
        "n_hypo_params": 8,
        "n_hit_features": 10,
        "n_evt_features": 2,
        "batch_size": {"n_hypos": 200, "n_observations": 6000},
        "send_hwm": 10000,
        "recv_hwm": 10000,
        "hitnet_file": hitnet,
        "chargenet_file": chargenet,
        "boundary_guard": {
            "file": theta_prior,
            "param_limits": DEFAULT_SEARCH_LIMITS,
            "bg_lim": -10,
            "invalid_llh": 1e9,
            "prior": True,
            "Tprior": t_prior,
        },
    }


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--hitnet",
        type=str,
        default="/home/atfienberg/freedomDataCopy/resources/HitNet_ranger_total_03_Jan_2021-16h05/epoch_48_model.hdf5",
        help="""hitnet file path""",
    )
    parser.add_argument(
        "--chargenet",
        type=str,
        default="/home/atfienberg/freedomDataCopy/resources/ChargeNet_nChannels_22_May_2020-11h05/epoch_400_model.hdf5",
        help="""chargenet file path""",
    )
    parser.add_argument(
        "--theta_prior",
        type=str,
        default="/home/atfienberg/IceCube/freeDOM/freedom/resources/prior/oscNext_theta_prior_norm.hdf5",
        help="""theta prior file path""",
    )
    parser.add_argument(
        "--t_prior",
        type=str,
        default="/home/atfienberg/IceCube/freeDOM/freedom/resources/prior/oscNext_time_residual_prior.pkl",
        help="""time prior file path""",
    )
    parser.add_argument(
        "--cuda_device", type=int, required=True, help="""cuda device index"""
    )
    parser.add_argument(
        "--kill", action="store_true", help="""kill a running LLH service"""
    )

    args = parser.parse_args()

    adj_ctrl_addr = adjust_addr_string(ctrl_addr, args.cuda_device)
    adj_req_addr = adjust_addr_string(req_addr, args.cuda_device)

    service_conf = build_service_conf(
        args.hitnet, args.chargenet, args.theta_prior, args.t_prior
    )

    if args.kill:
        print(f"Killing service at {adj_ctrl_addr}")
        kill_service(adj_ctrl_addr)
    else:
        print(f"Starting service at {adj_ctrl_addr}")

        start_service(
            service_conf,
            ctrl_addr=adj_ctrl_addr,
            req_addr=adj_req_addr,
            cuda_device=args.cuda_device,
        )


if __name__ == "__main__":
    main()
