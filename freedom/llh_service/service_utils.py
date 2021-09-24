"""LLH service utility functions"""
import zmq


def kill_service(ctrl_addr):
    """kill the LLH service running at `ctrl_addr`"""
    with zmq.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(ctrl_addr)

        sock.send_string("die")

        return sock.recv_string()


def set_service_environ(cuda_device):
    import os
    import tensorflow as tf

    # use a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def start_service(params, ctrl_addr, req_addr, cuda_device):
    set_service_environ(cuda_device)
    from freedom.llh_service.llh_service import LLHService

    params = params.copy()
    params["ctrl_addr"] = ctrl_addr
    params["req_addr"] = req_addr

    with LLHService(**params) as serv:
        print(
            f"starting service work loop for cuda device {cuda_device} at ctrl_addr {serv.ctrl_addr}",
            flush=True,
        )
        serv.start_work_loop()


def start_service_pipe(service_conf, cuda_device, pipe):
    """start an LLHService and send its ctrl addr through `pipe`"""
    set_service_environ(cuda_device)
    from freedom.llh_service.llh_service import LLHService

    with LLHService(**service_conf) as serv:
        pipe.send(serv.ctrl_addr)
        serv.start_work_loop()
