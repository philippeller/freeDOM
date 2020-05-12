#!/usr/bin/env python

"""
llh service:
listens for llh evaluation requests and processes them in batches
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import json
import os
import time
import sys

import numpy as np
import numba
import tensorflow as tf
import zmq

FREEDOM_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if FREEDOM_DIR not in sys.path:
    sys.path.append(FREEDOM_DIR)
from freeDOM.transformations import chargenet_trafo, hitnet_trafo
import llh_cython
import eval_llh


def wstdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()


# fake eval_llh for development
zero_floats = np.full(10000, 10, np.float32)


class fake_llh:
    def __init__(self):
        pass

    def numpy(self):
        return zero_floats


def fake_eval_llh(*args, **kwargs):
    return fake_llh()


class LLHService:

    __slots__ = [
        "_eval_llh",
        "_work_reqs",
        "_n_table_rows",
        "_n_obs_features",
        "_n_hypos",
        "_n_hypo_params",
        "_x_table",
        "_theta_table",
        "_stop_inds",
        "_next_table_ind",
        "_next_hypo_ind",
        "_model",
        "_flush_period",
        "_poll_timeout",
        "_ctxt",
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
        n_hypo_params,
        n_obs_features,
        batch_size,
        send_hwm,
        recv_hwm,
        transform_params=None,
        use_freeDOM_model=False,
        hitnet_file=None,
        chargenet_file=None,
        router_mandatory=False,
        bypass_tensorflow=False,
    ):
        self._work_reqs = []

        self._n_table_rows = batch_size["n_observations"]

        self._n_hypos = batch_size["n_hypos"]
        """number of hypotheses per batch"""

        self._n_obs_features = n_obs_features
        self._n_hypo_params = n_hypo_params

        self._x_table = np.zeros(
            (self._n_table_rows, self._n_obs_features), dtype=np.float32
        )
        self._theta_table = np.zeros(
            (self._n_hypos, self._n_hypo_params), dtype=np.float32
        )
        self._stop_inds = np.full(
            shape=(self._n_hypos,), fill_value=self._n_table_rows, dtype=np.int32
        )
        self._next_table_ind = 0
        self._next_hypo_ind = 0

        if not use_freeDOM_model:
            classifier = tf.keras.models.load_model(model_file)

            # build a model that includes the normalization
            self._model = eval_llh.build_norm_model(classifier, **transform_params)

            self._eval_llh = eval_llh.eval_llh
        else:
            hitnet = tf.keras.models.load_model(
                hitnet_file, custom_objects={"hitnet_trafo": hitnet_trafo}
            )
            chargenet = tf.keras.models.load_model(
                chargenet_file, custom_objects={"chargenet_trafo": chargenet_trafo}
            )
            self._model = (hitnet, chargenet)

            self._eval_llh = eval_llh.freedom_nllh

        if bypass_tensorflow:
            self._eval_llh = fake_eval_llh

        # trace-compile the llh function in advance
        self._eval_llh(
            tf.constant(self._x_table),
            tf.constant(self._theta_table),
            tf.constant(self._stop_inds),
            self._model,
        )

        # convert flush period to seconds
        self._flush_period = flush_period / 1000.0

        self._poll_timeout = poll_timeout

        self._ctxt = None
        self._req_sock = None
        self._ctrl_sock = None
        self._last_flush = 0

        self._init_sockets(
            req_addr=req_addr,
            ctrl_addr=ctrl_addr,
            send_hwm=send_hwm,
            recv_hwm=recv_hwm,
            router_mandatory=router_mandatory,
        )

        # trace-compile the llh function in advance
        self._eval_llh(
            tf.constant(self._x_table),
            tf.constant(self._theta_table),
            tf.constant(self._stop_inds),
            self._model,
        )

        # # jit compile self._fill_tables
        # self._fill_tables(
        #     self._x_table,
        #     self._theta_table,
        #     self._stop_inds,
        #     np.zeros(self._n_obs_features, np.float32),
        #     np.zeros(self._n_hypo_params, np.float32),
        #     0,
        #     0,
        # )
        # self._flush()

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

            for sock, _ in events:
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

    def _init_sockets(self, req_addr, ctrl_addr, send_hwm, recv_hwm, router_mandatory):
        # pylint: disable=no-member
        self._ctxt = zmq.Context()

        req_sock = self._ctxt.socket(zmq.ROUTER)
        req_sock.setsockopt(zmq.SNDHWM, send_hwm)
        req_sock.setsockopt(zmq.RCVHWM, recv_hwm)
        if router_mandatory:
            req_sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        req_sock.bind(req_addr)

        ctrl_sock = self._ctxt.socket(zmq.PULL)
        ctrl_sock.bind(ctrl_addr)

        self._req_sock = req_sock
        self._ctrl_sock = ctrl_sock

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print("cleaning up")
        self._ctxt.destroy()

    def _process_message(self, msg_parts):
        # wstdout(".")
        header_frames = msg_parts[:-2]
        x, theta = msg_parts[-2:]

        x = np.frombuffer(x, np.float32)
        n_obs = int(len(x) / self._n_obs_features)
        x = x.reshape(n_obs, self._n_obs_features)

        thetas = np.frombuffer(theta, np.float32)
        batch_size = int(len(thetas) / self._n_hypo_params)
        thetas = thetas.reshape(batch_size, self._n_hypo_params)

        next_ind = self._next_table_ind
        hypo_ind = self._next_hypo_ind

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

        self._record_req(
            x, thetas, next_ind, hypo_ind, stop_ind, stop_hypo_ind, header_frames
        )
        # self._numba_record_req(x, thetas, next_ind, hypo_ind, header_frames)

    # @profile
    def _record_req(
        self, x, thetas, next_ind, hypo_ind, stop_ind, stop_hypo_ind, header_frames
    ):
        batch_size = len(thetas)
        n_obs = len(x)

        # fill table with observations and hypothesis parameters
        self._x_table[next_ind:stop_ind] = np.tile(x, (batch_size, 1))
        self._theta_table[hypo_ind:stop_hypo_ind] = thetas

        # update stop indices
        next_stop = next_ind + n_obs
        self._stop_inds[hypo_ind : hypo_ind + batch_size] = np.arange(
            next_stop, next_stop + n_obs * batch_size, n_obs
        )

        # record work request information
        work_item_dict = dict(
            header_frames=header_frames, start_ind=hypo_ind, stop_ind=stop_hypo_ind,
        )

        self._work_reqs.append(work_item_dict)
        self._next_table_ind = stop_ind
        self._next_hypo_ind = stop_hypo_ind

    # @profile
    def _numba_record_req(self, x, thetas, next_ind, hypo_ind, header_frames):
        self._fill_tables(
            self._x_table,
            self._theta_table,
            self._stop_inds,
            x,
            thetas,
            next_ind,
            hypo_ind,
        )

        stop_hypo_ind = hypo_ind + len(thetas)

        # record work request information
        work_item_dict = dict(
            header_frames=header_frames, start_ind=hypo_ind, stop_ind=stop_hypo_ind,
        )
        self._work_reqs.append(work_item_dict)
        self._next_table_ind = next_ind + len(x) * len(thetas)
        self._next_hypo_ind = stop_hypo_ind

    @staticmethod
    @numba.njit
    def _fill_tables(x_table, theta_table, stop_inds, x, thetas, next_ind, hypo_ind):

        batch_size = len(thetas)
        n_obs = len(x)

        # fill table with observations and hypothesis parameters
        for i in range(batch_size):
            start = next_ind + i * n_obs
            stop = start + n_obs
            x_table[start:stop] = x

        theta_table[hypo_ind : hypo_ind + batch_size] = thetas

        # update stop indices
        next_stop = next_ind + n_obs
        stop_inds[hypo_ind : hypo_ind + batch_size] = np.arange(
            next_stop, next_stop + n_obs * batch_size, n_obs
        )

    # @profile
    def _process_all_reqs(self):
        # pylint: disable=no-member
        while True:
            try:
                # frames = self._req_sock.recv_multipart(zmq.DONTWAIT)
                frames = llh_cython.receive_req(self._req_sock)
                self._process_message(frames)
            except zmq.error.Again:
                # no more messages
                return

    def _process_ctrl_cmd(self):
        """read a message from the ctrl socket
        currently the only valid control command is "die",
        which commands the service to exit the work loop.

        Could become more complicated later
        """
        # pylint: disable=no-member
        try:
            return self._ctrl_sock.recv_string(zmq.DONTWAIT)
        except zmq.error.Again:
            # this should never happen, we are receiving only after polling.
            # print a message and raise again
            print(
                "Failed to receive from ctrl sock even after"
                " the poller indicated an event was ready!"
            )
            raise

    def _flush(self):
        self._last_flush = time.time()
        # wstdout("F")

        if self._work_reqs:
            # wstdout("+")
            x_table = tf.constant(self._x_table)
            theta_table = tf.constant(self._theta_table)
            stop_inds = tf.constant(self._stop_inds)
            llh_sums = self._eval_llh(x_table, theta_table, stop_inds, self._model)
            llhs = llh_sums.numpy()

            # self._dispatch_replies(llhs)
            llh_cython.dispatch_replies(self._req_sock, self._work_reqs, llhs)

            self._work_reqs.clear()
            self._next_table_ind = 0
            self._next_hypo_ind = 0
            self._stop_inds[:] = self._n_table_rows

    # @profile
    def _dispatch_replies(self, llhs):
        for work_req in self._work_reqs:
            llh_slice = llhs[work_req["start_ind"] : work_req["stop_ind"]]
            frames = work_req["header_frames"] + [llh_slice]
            self._req_sock.send_multipart(frames)


def main():
    with open("service_params.json") as f:
        params = json.load(f)

    with LLHService(**params) as service:
        wstdout("starting work loop:\n")
        service.start_work_loop()


if __name__ == "__main__":
    sys.exit(main())
