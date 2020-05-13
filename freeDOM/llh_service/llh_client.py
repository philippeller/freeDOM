"""
llh client:
packages messages, sends them to the llh service, and interprets replies
provides synchronous and asynchronous interfaces
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import uuid

import numpy as np
import zmq

import llh_cython


class LLHClient:

    slots = [
        "_max_hypos_per_batch",
        "_max_obs_per_batch",
        "_n_hypo_params",
        "_n_obs_features",
        "_sock",
    ]

    def __init__(self, req_addr, batch_size, n_hypo_params, n_obs_features):
        self._init_socket(req_addr)
        self._max_hypos_per_batch = batch_size["n_hypos"]
        self._max_obs_per_batch = batch_size["n_observations"]
        self._n_hypo_params = n_hypo_params
        self._n_obs_features = n_obs_features

    @property
    def max_obs_per_batch(self):
        return self._max_obs_per_batch

    @property
    def max_hypos_per_batch(self):
        return self._max_hypos_per_batch

    def request_eval(self, x, theta, req_id=""):
        """Request a single llh eval

        Parameters
        ----------
        x : observations: numpy.ndarray of dtype float32
        theta: hypothesis params: numpy.ndarray of dtype float32
        req_id : optional
            Converted to str, and returned as such

        """

        n_obs = x.size / self._n_obs_features

        if n_obs > self._max_obs_per_batch:
            raise ValueError(
                "x.size / n_obs_features must be <= the maximum batch size!"
                f" (In this case {self._max_obs_per_batch})"
            )

        # send a req_id string for development and debugging
        req_id_bytes = str(req_id).encode()

        # self._sock.send_multipart([req_id_bytes, x, theta])
        llh_cython.dispatch_request(self._sock, req_id_bytes, x, theta)

    def request_batch_eval(self, x, thetas, req_id=""):
        """Request batch eval of llh(x|mu, sig) for all mus and sigs

        Parameters
        ----------
        x : observations numpy.ndarray of dtype float32
        thetas: hypothesis parameters to evaluate, numpy.ndarray of dtype float32
        req_id : optional
            Converted to a string and returned as such
        """

        n_obs = x.size / self._n_obs_features
        n_hypos = thetas.size / self._n_hypo_params

        if n_obs * n_hypos > self._max_obs_per_batch:
            raise ValueError(
                "n_obs*n_hypos must be <= the maximum batch size!"
                f" (In this case {self._max_obs_per_batch})"
            )

        if n_hypos > self._max_hypos_per_batch:
            raise ValueError(
                "n_hypos must be <= the maximum hypothesis batch size!"
                f" (In this case {self._max_hypos_per_batch})"
            )

        req_id_bytes = str(req_id).encode()

        # self._sock.send_multipart([req_id_bytes, x, thetas])
        llh_cython.dispatch_request(self._sock, req_id_bytes, x, thetas)

    def recv(self, timeout=None):
        if self._sock.poll(timeout, zmq.POLLIN) != 0:
            req_id, llh = self._sock.recv_multipart()
            return dict(req_id=req_id.decode(), llh=np.frombuffer(llh, np.float32))
        return None

    def eval_llh(self, x, theta, timeout=None):
        """Synchronous llh evaluation, blocking until llh is ready.

        .. warning:: Do not use while asynchronous requests are in progress.

        Parameters
        ----------
        x : numpy.ndarray of dtype float32
            Observations
        theta: numpy.ndarray of dtype float32
            Hypothesis parameters
        timeout : int, optional
            Wait for a reply up to `timeout` milliseconds
        max_retries : int >= 0, optional
            Retry up to this many times if `timeout` occurs or an empty reply

        Raises
        ------
        RuntimeError
            On reaching timeout or failure of internal message uuid check
        """

        req_id = uuid.uuid4().hex

        self.request_eval(x, theta, req_id=req_id)

        reply = self.recv(timeout)
        if reply is None:
            raise RuntimeError("No reply from LLH service!")

        if reply["req_id"] != req_id:
            raise RuntimeError("uuid mismatch!")

        return reply["llh"][0]

    def _init_socket(self, req_addr):
        # pylint: disable=no-member
        ctxt = zmq.Context.instance()
        sock = ctxt.socket(zmq.DEALER)
        sock.connect(req_addr)

        self._sock = sock
        self._sock.setsockopt(zmq.LINGER, 0)
