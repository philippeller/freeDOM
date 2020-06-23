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
import pkg_resources

import numpy as np

# import numba
import tensorflow as tf
import zmq

from freedom.neural_nets.transformations import chargenet_trafo, hitnet_trafo
from freedom.neural_nets.transformations import stringnet_trafo, layernet_trafo
from freedom.llh_service import llh_cython
from freedom.llh_service import eval_llh


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
        "_n_hit_features",
        "_n_evt_features",
        "_n_hypos",
        "_n_hypo_params",
        "_hit_table",
        "_evt_data_table",
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
        "_client_conf",
    ]

    def __init__(
        self,
        req_addr,
        ctrl_addr,
        poll_timeout,
        flush_period,
        n_hypo_params,
        n_hit_features,
        n_evt_features,
        batch_size,
        send_hwm,
        recv_hwm,
        transform_params=None,
        use_freeDOM_model=True,
        model_file=None,
        hitnet_file=None,
        chargenet_file=None,
        stringnet_file=None,
        layernet_file=None,
        router_mandatory=False,
        bypass_tensorflow=False,
        n_strings=86,
        features_per_string=5,
        n_layers=60,
        features_per_layer=4,
    ):
        if (chargenet_file is None) + (stringnet_file is None) + (
            layernet_file is None
        ) != 2:
            raise RuntimeError(
                "You must select exactly one of chargenet, stringnet, or layernet."
            )

        self._work_reqs = []

        self._n_table_rows = batch_size["n_observations"]

        self._n_hypos = batch_size["n_hypos"]
        """number of hypotheses per batch"""

        self._n_hit_features = n_hit_features
        self._n_evt_features = n_evt_features
        self._n_hypo_params = n_hypo_params

        self._hit_table = np.zeros(
            (self._n_table_rows, self._n_hit_features), dtype=np.float32
        )
        self._evt_data_table = np.zeros(
            (self._n_hypos, self._n_evt_features), dtype=np.float32
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
            hitnet_file = self._get_model_path(hitnet_file)
            hitnet = tf.keras.models.load_model(
                hitnet_file, custom_objects={"hitnet_trafo": hitnet_trafo}
            )
            hitnet.layers[-1].activation = tf.keras.activations.linear

            if chargenet_file is not None:
                chargenet_file = self._get_model_path(chargenet_file)
                chargenet = tf.keras.models.load_model(
                    chargenet_file, custom_objects={"chargenet_trafo": chargenet_trafo}
                )
                chargenet.layers[-1].activation = tf.keras.activations.linear

            elif stringnet_file is not None:
                stringnet_file = self._get_model_path(stringnet_file)
                stringnet = tf.keras.models.load_model(
                    stringnet_file, custom_objects={"stringnet_trafo": stringnet_trafo}
                )
                stringnet.layers[-1].activation = tf.keras.activations.linear

                chargenet = eval_llh.wrap_partial_chargenet(
                    stringnet, n_hypo_params, n_strings, features_per_string
                )

            elif layernet_file is not None:
                layernet_file = self._get_model_path(layernet_file)
                layernet = tf.keras.models.load_model(
                    layernet_file, custom_objects={"layernet_trafo": layernet_trafo}
                )
                layernet.layers[-1].activation = tf.keras.activations.linear

                chargenet = eval_llh.wrap_partial_chargenet(
                    layernet, n_hypo_params, n_layers, features_per_layer
                )

            self._model = (hitnet, chargenet)

            self._eval_llh = eval_llh.freedom_nllh

        if bypass_tensorflow:
            self._eval_llh = fake_eval_llh

        # trace-compile the llh function in advance
        self._eval_llh(
            tf.constant(self._hit_table),
            tf.constant(self._evt_data_table),
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

        # store configuration info for clients
        self._client_conf = dict(
            batch_size=batch_size,
            n_hypo_params=n_hypo_params,
            n_hit_features=n_hit_features,
            n_evt_features=n_evt_features,
            req_addr=req_addr,
        )

        # # jit compile self._fill_tables
        # self._fill_tables(
        #     self._hit_table,
        #     self._theta_table,
        #     self._stop_inds,
        #     np.zeros(self._n_hit_features, np.float32),
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
                    self._process_reqs()
                elif sock is self._ctrl_sock:
                    cmd = self._process_ctrl_cmd()
                    if cmd == "die":
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

        ctrl_sock = self._ctxt.socket(zmq.REP)
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
        header_frames = msg_parts[:-3]
        hit_data, evt_data, theta = msg_parts[-3:]

        hit_data = np.frombuffer(hit_data, np.float32)
        n_hits = int(len(hit_data) / self._n_hit_features)
        hit_data = hit_data.reshape(n_hits, self._n_hit_features)

        evt_data = np.frombuffer(evt_data, np.float32)
        # to-do: potentially add better error checking here.
        # for now, message will be discarded if shape of evt_data is wrong
        if evt_data.size != self._n_evt_features:
            return

        thetas = np.frombuffer(theta, np.float32)
        batch_size = int(len(thetas) / self._n_hypo_params)
        thetas = thetas.reshape(batch_size, self._n_hypo_params)

        next_ind = self._next_table_ind
        hypo_ind = self._next_hypo_ind

        n_rows = n_hits * batch_size

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
            hit_data,
            evt_data,
            thetas,
            next_ind,
            hypo_ind,
            stop_ind,
            stop_hypo_ind,
            header_frames,
        )
        # self._numba_record_req(x, thetas, next_ind, hypo_ind, header_frames)

    # @profile
    def _record_req(
        self,
        hit_data,
        evt_data,
        thetas,
        next_ind,
        hypo_ind,
        stop_ind,
        stop_hypo_ind,
        header_frames,
    ):
        batch_size = len(thetas)
        n_hits = len(hit_data)

        # fill tables with data and hypothesis parameters
        self._hit_table[next_ind:stop_ind] = np.tile(hit_data, (batch_size, 1))
        self._evt_data_table[hypo_ind:stop_hypo_ind] = evt_data
        self._theta_table[hypo_ind:stop_hypo_ind] = thetas

        # update stop indices
        next_stop = next_ind + n_hits
        self._stop_inds[hypo_ind : hypo_ind + batch_size] = np.arange(
            next_stop, next_stop + n_hits * batch_size, n_hits
        )

        # record work request information
        work_item_dict = dict(
            header_frames=header_frames, start_ind=hypo_ind, stop_ind=stop_hypo_ind,
        )

        self._work_reqs.append(work_item_dict)
        self._next_table_ind = stop_ind
        self._next_hypo_ind = stop_hypo_ind

    #     # @profile
    #     def _numba_record_req(self, x, thetas, next_ind, hypo_ind, header_frames):
    #         self._fill_tables(
    #             self._hit_table,
    #             self._theta_table,
    #             self._stop_inds,
    #             x,
    #             thetas,
    #             next_ind,
    #             hypo_ind,
    #         )

    #         stop_hypo_ind = hypo_ind + len(thetas)

    #         # record work request information
    #         work_item_dict = dict(
    #             header_frames=header_frames, start_ind=hypo_ind, stop_ind=stop_hypo_ind,
    #         )
    #         self._work_reqs.append(work_item_dict)
    #         self._next_table_ind = next_ind + len(x) * len(thetas)
    #         self._next_hypo_ind = stop_hypo_ind

    #     @staticmethod
    #     @numba.njit
    #     def _fill_tables(x_table, theta_table, stop_inds, x, thetas, next_ind, hypo_ind):

    #         batch_size = len(thetas)
    #         n_obs = len(x)

    #         # fill table with observations and hypothesis parameters
    #         for i in range(batch_size):
    #             start = next_ind + i * n_obs
    #             stop = start + n_obs
    #             x_table[start:stop] = x

    #         theta_table[hypo_ind : hypo_ind + batch_size] = thetas

    #         # update stop indices
    #         next_stop = next_ind + n_obs
    #         stop_inds[hypo_ind : hypo_ind + batch_size] = np.arange(
    #             next_stop, next_stop + n_obs * batch_size, n_obs
    #         )

    # @profile
    def _process_reqs(self):
        """
        process up to n_hypos requests before yielding control
        """
        # pylint: disable=no-member
        for _ in range(self._n_hypos):
            try:
                # frames = self._req_sock.recv_multipart(zmq.DONTWAIT)
                frames = llh_cython.receive_req(self._req_sock)
                self._process_message(frames)
            except zmq.error.Again:
                # no more messages
                return

    def _process_ctrl_cmd(self):
        """read a message from the ctrl socket
        currently the only valid control commands are "die",
        which commands the service to exit the work loop,
        and "conf", which sends the client conf

        Could become more complicated later
        """
        # pylint: disable=no-member
        try:
            cmd = self._ctrl_sock.recv_string(zmq.DONTWAIT)
        except zmq.error.Again:
            # this should never happen, we are receiving only after polling.
            # print a message and raise again
            print(
                "Failed to receive from ctrl sock even after"
                " the poller indicated an event was ready!"
            )
            raise

        if cmd == "die":
            self._ctrl_sock.send_string("dying")
        elif cmd == "conf":
            self._ctrl_sock.send_json(self._client_conf)
        else:
            self._ctrl_sock.send_string("?")

        return cmd

    def _flush(self):
        self._last_flush = time.time()
        # wstdout("F")

        if self._work_reqs:
            # wstdout("+")
            hit_data_table = tf.constant(self._hit_table)
            evt_data_table = tf.constant(self._evt_data_table)
            theta_table = tf.constant(self._theta_table)
            stop_inds = tf.constant(self._stop_inds)
            llh_sums = self._eval_llh(
                hit_data_table, evt_data_table, theta_table, stop_inds, self._model
            )
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

    @staticmethod
    def _get_model_path(filename):
        if not os.path.exists(filename):
            filename = f"{pkg_resources.resource_filename('freedom', 'resources/models')}/{filename}"

        return filename
