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

from freeDOM.llh_service import llh_cython


class LLHClient:

    slots = [
        "_max_hypos_per_batch",
        "_max_obs_per_batch",
        "_n_hypo_params",
        "_n_obs_features",
        "_sock",
    ]

    def __init__(self, ctrl_addr, conf_timeout=-1):
        """ loads configuration info from service listening 
        at ctrl_addr """
        with zmq.Context.instance().socket(zmq.REQ) as req_sock:
            req_sock.setsockopt(zmq.LINGER, 0)
            req_sock.setsockopt(zmq.RCVTIMEO, conf_timeout)
            req_sock.connect(ctrl_addr)

            req_sock.send_string("conf")
            conf = req_sock.recv_json()

            self._init_from_conf(**conf)

    @property
    def max_obs_per_batch(self):
        return self._max_obs_per_batch

    @property
    def max_hypos_per_batch(self):
        return self._max_hypos_per_batch

    def request_eval(self, x, theta, req_id=""):
        """Asynchronous llh eval request

        Note: Batch sizes for asynchronous requests are currently limited to max_hypos_per_batch
        Synchronous requests are unlimited in size (see eval_llh)

        Replies to asynchronous requests are retrieved using LLHClient.recv()

        Parameters
        ----------
        x : observations: numpy.ndarray, or something convertible to one  
        theta: hypothesis params: numpy.ndarray, or something convertible to one 
        req_id : optional
            Converted to str, and returned as such

        """
        x, theta = self._prepare_and_check_buffers(x, theta)

        n_hypos = theta.size / self._n_hypo_params
        n_obs = x.size / self._n_obs_features

        if n_hypos > self._max_hypos_per_batch:
            raise RuntimeError(
                f"Asynchronous requests are limited to {self.max_hypos_per_batch} hypotheses "
                f"per req, but {n_hypos:.0f} were requested!"
            )

        if n_obs * n_hypos > self._max_obs_per_batch:
            raise RuntimeError(
                f"Asynchronous requests are limited to {self._max_obs_per_batch} total pulses "
                f"per req, but {n_obs*n_hypos:.0f} were requested!"
            )

        # send a req_id string for development and debugging
        req_id_bytes = str(req_id).encode()

        # self._sock.send_multipart([req_id_bytes, x, theta])
        llh_cython.dispatch_request(self._sock, req_id_bytes, x, theta)

    def eval_llh(self, x, theta, timeout=None):
        """Synchronous llh evaluation, blocking until llh is ready.

        Batch size is unlimited for synchronous requests
        (although it may take a while to get your reply)

        .. warning:: Do not use while asynchronous requests are in progress.

        Parameters
        ----------
        x : numpy.ndarray, or something convertible to one  
            Observations
        theta: numpy.ndarray, or something convertible to one  
            Hypothesis parameters
        timeout : int, optional
            Wait for a reply up to `timeout` milliseconds

        Raises
        ------
        RuntimeError
            On reaching timeout or failure of internal message uuid check
        """

        x, theta = self._prepare_and_check_buffers(x, theta)

        n_hypos = theta.size / self._n_hypo_params
        n_obs = x.size / self._n_obs_features

        # split into multiple requests if necessary
        if (
            n_hypos > self._max_hypos_per_batch
            or n_obs * n_hypos > self._max_obs_per_batch
        ):
            if n_obs > self._max_obs_per_batch:
                raise ValueError(
                    "Current LLH service only supports events with up to "
                    f"{self._max_obs_per_batch} pulses!"
                )

            if n_obs * self._max_hypos_per_batch <= self._max_obs_per_batch:
                hypos_per_split = self._max_hypos_per_batch
            else:
                hypos_per_split = int(self._max_obs_per_batch / n_obs)

            split_step = hypos_per_split * self._n_hypo_params

            theta_splits = np.split(
                theta, np.arange(split_step, theta.size, split_step)
            )

        else:
            theta_splits = [theta]

        req_ids = []
        for theta_split in theta_splits:
            req_id = uuid.uuid4().hex
            req_ids.append(req_id)

            self.request_eval(x, theta_split, req_id=req_id)

        llhs = np.empty(int(n_hypos), dtype=np.float32)
        llh_view = llhs
        for req_id in req_ids:
            reply = self.recv(timeout)
            if reply is None:
                raise RuntimeError("No reply from LLH service!")

            if reply["req_id"] != req_id:
                raise RuntimeError("uuid mismatch!")

            llh_split = reply["llh"]

            llh_view[: llh_split.size] = llh_split
            llh_view = llh_view[llh_split.size :]

        assert llh_view.size == 0

        if n_hypos == 1:
            return llhs[0]
        else:
            return llhs

    def recv(self, timeout=None):
        """
        attempt to retrieve a reply from the LLH service
        returns None if no reply is available

        Parameters
        ----------
        timeout : int, optional
            Wait for a reply up to `timeout` milliseconds
        """
        if self._sock.poll(timeout, zmq.POLLIN) != 0:
            req_id, llh = self._sock.recv_multipart()
            return dict(req_id=req_id.decode(), llh=np.frombuffer(llh, np.float32))
        return None

    def _init_socket(self, req_addr):
        # pylint: disable=no-member
        ctxt = zmq.Context.instance()
        sock = ctxt.socket(zmq.DEALER)
        sock.connect(req_addr)

        self._sock = sock
        self._sock.setsockopt(zmq.LINGER, 0)

    def _init_from_conf(self, req_addr, batch_size, n_hypo_params, n_obs_features):
        self._init_socket(req_addr)
        self._max_hypos_per_batch = batch_size["n_hypos"]
        self._max_obs_per_batch = batch_size["n_observations"]
        self._n_hypo_params = n_hypo_params
        self._n_obs_features = n_obs_features

    def _prepare_and_check_buffers(self, x, theta):
        """ validates x & theta. Converts them to contiguous, flat arrays of type np.float32 if they are not already """

        x, theta = (self._as_flat_float_array(arr) for arr in (x, theta))

        if x.size % self._n_obs_features != 0:
            raise ValueError(
                f"x.size must be divisible by the number of observation features ({self._n_obs_features})"
            )

        if theta.size % self._n_hypo_params != 0:
            raise ValueError(
                f"theta.size must be divisible by the number of hypothesis parameters ({self._n_hypo_params})"
            )

        return x.reshape(x.size,), theta.reshape(theta.size,)

    @staticmethod
    def _as_flat_float_array(arr):
        if (
            not isinstance(arr, np.ndarray)
            or not arr.flags.c_contiguous
            or arr.dtype != np.float32
        ):
            arr = np.ascontiguousarray(arr, dtype=np.float32)

        return arr.reshape(arr.size,)
