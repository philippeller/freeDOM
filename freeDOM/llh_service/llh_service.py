"""
llh service:
listens for llh evaluation requests and processes them in batches
"""


from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import json
import time
import sys

import numpy as np
import tensorflow as tf
import zmq

import eval_llh


class LLHService:

    __slots__ = [
        "_eval_llh",
        "_work_reqs",
        "_n_table_rows",
        "_n_hypos",
        "_n_hypo_params",
        "_table",
        "_stop_inds",
        "_next_table_ind",
        "_next_hypo_ind",
        "_model",
        "_eval_llh",
        "_flush_period",
        "_poll_timeout",
        "_req_sock",
        "_ctrl_sock",
        "_last_flush",
    ]

    def __init__(
        self,
        req_addr,
        ctrl_addr,
        poll_timeout,
        flush_period,
        model_file,
        n_features,
        batch_size,
        transform_params,
    ):
        self._eval_llh = eval_llh.eval_llh

        self._work_reqs = []

        self._n_table_rows = batch_size["n_observations"]

        self._n_hypos = batch_size["n_hypos"]
        """number of hypotheses per batch"""

        # note: my example has one "observation" feature,
        # yielding the -1 below. This should be made more general
        self._n_hypo_params = n_features - 1

        self._table = np.zeros((self._n_table_rows, n_features), dtype=np.float32)

        self._stop_inds = np.full(self._n_hypos + 1, self._n_table_rows, np.int32)
        self._next_table_ind = 0
        self._next_hypo_ind = 0

        classifier = tf.keras.models.load_model(model_file)

        # build a model that includes the normalization
        self._model = eval_llh.build_norm_model(classifier, **transform_params)

        # trace-compile the llh function in advance
        self._eval_llh(
            tf.constant(self._table, tf.float32),
            tf.constant(self._stop_inds, tf.int32),
            self._model,
        )

        # convert flush period to seconds
        self._flush_period = flush_period / 1000.0

        self._poll_timeout = poll_timeout

        self._req_sock = None
        self._ctrl_sock = None
        self._last_flush = 0

        self._init_sockets(req_addr=req_addr, ctrl_addr=ctrl_addr)

    # @profile
    def start_work_loop(self):
        flush_period = self._flush_period
        self._last_flush = time.time()

        poll_timeout = self._poll_timeout

        poller = zmq.Poller()
        poller.register(self._req_sock, zmq.POLLIN)
        poller.register(self._ctrl_sock, zmq.POLLIN)

        while True:
            events = poller.poll(poll_timeout)

            for sock, evt in events:
                if sock is self._req_sock:
                    self._process_all_reqs()
                elif sock is self._ctrl_sock:
                    action = self._process_ctrl_cmd()
                    if action == "die":
                        print("Received die command... flushing and exiting")
                        self._flush()
                        return

            if time.time() - self._last_flush > flush_period:
                self._flush()

    def _init_sockets(self, req_addr, ctrl_addr):
        ctxt = zmq.Context.instance()
        req_sock = ctxt.socket(zmq.ROUTER)
        req_sock.bind(req_addr)

        ctrl_sock = ctxt.socket(zmq.PULL)
        ctrl_sock.bind(ctrl_addr)

        self._req_sock = req_sock
        self._ctrl_sock = ctrl_sock

    # @profile
    def _process_message(self, msg_parts):
        header_frames = msg_parts[:-2]
        x, theta = msg_parts[-2:]

        x = np.frombuffer(x, np.float32)

        thetas = np.frombuffer(theta, np.float32)

        next_ind = self._next_table_ind

        n_obs = len(x)
        hypo_ind = self._next_hypo_ind
        batch_size = int(len(thetas) / self._n_hypo_params)

        thetas = thetas.reshape(batch_size, self._n_hypo_params)

        n_rows = n_obs * batch_size

        # to-do: add better error checking
        # for now, message will be ignored if batch size is too large
        if n_rows > self._n_table_rows or batch_size > self._n_hypos:
            return

        # indices into _table and _stop_inds
        stop_ind = next_ind + n_rows
        stop_hypo_ind = hypo_ind + batch_size
        if stop_ind > self._n_table_rows or stop_hypo_ind > self._n_hypos:
            self._flush()
            next_ind = 0
            hypo_ind = 0
            stop_ind = n_rows
            stop_hypo_ind = batch_size

        # fill table with observations and hypothesis parameters
        table = self._table
        table[next_ind:stop_ind, 0] = np.tile(x, batch_size)
        for i in range(self._n_hypo_params):
            table[next_ind:stop_ind, i + 1] = np.repeat(thetas[:, i], n_obs)

        # update stop indices
        next_stop = next_ind + n_obs
        for i in range(hypo_ind, hypo_ind + batch_size):
            self._stop_inds[i] = next_stop
            next_stop += n_obs

        # record work request information
        work_item_dict = {}

        work_item_dict["header_frames"] = header_frames
        work_item_dict["start_ind"] = hypo_ind
        work_item_dict["stop_ind"] = stop_hypo_ind

        self._work_reqs.append(work_item_dict)
        self._next_table_ind = stop_ind
        self._next_hypo_ind = stop_hypo_ind

    def _process_all_reqs(self):
        while True:
            try:
                self._process_message(self._req_sock.recv_multipart(zmq.NOBLOCK))
            except zmq.error.Again:
                # no more messages
                return

    def _process_ctrl_cmd(self):
        """read a message from the ctrl socket
        currently the only valid control command is "die",
        which commands the service to exit the work loop.

        Could become more complicated later
        """
        try:
            return self._ctrl_sock.recv_string(zmq.NOBLOCK)
        except zmq.error.Again:
            """ this should never happen, we are receiving only after polling.
            print a message and raise again
            """
            print(
                "Failed to receive from ctrl sock even after"
                " the poller indicated an event was ready!"
            )
            raise

    def _flush(self):
        start = time.time()

        self._last_flush = time.time()

        work_reqs = self._work_reqs

        if len(work_reqs) > 0:
            llh_start = time.time()
            llh_func = self._eval_llh
            llh_sums = llh_func(
                tf.constant(self._table, tf.float32),
                tf.constant(self._stop_inds, tf.int32),
                self._model,
            )
            llhs = llh_sums.numpy()
            delta = time.time() - llh_start
            print(f"llh_eval took {delta*1000:.3f} ms")
        else:
            llhs = []

        for work_req in work_reqs:
            llh_slice = llhs[work_req["start_ind"] : work_req["stop_ind"]]

            self._req_sock.send_multipart(work_req["header_frames"] + [llh_slice])

        if len(work_reqs) > 0:
            stop = time.time()

            print(f"flush took {(stop-start)*1000:.1f} ms")

        self._work_reqs = []
        self._next_table_ind = 0
        self._stop_inds[:] = self._n_table_rows
        self._next_hypo_ind = 0


def main():
    with open("service_params.json") as f:
        params = json.load(f)

    service = LLHService(**params)

    print("starting work loop:")
    service.start_work_loop()


if __name__ == "__main__":
    sys.exit(main())
