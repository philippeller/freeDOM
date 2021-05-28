"""
cpu client:
Like the llh client, but calculates likelihoods itself rather than
shipping requests to the LLH Service. Can be used most places that
an llh client would, e.g. for an IceTray FreeDOM reco 

This is intended for running FreeDOM reconstructions on nodes with no
GPUs, motivating the name, although it can also use GPUs if they are
available.
"""

from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from freedom.neural_nets.transformations import chargenet_trafo, hitnet_trafo
from freedom.llh_service import eval_llh, llh_service


class CPUClient:

    slots = ["_models", "_batch_size", "_boundary_guard"]

    def __init__(self, hitnet_file, chargenet_file, batch_size=12, boundary_guard=None):
        """ loads models given file paths, configures boundary guard & eval batch size
        """
        hitnet = tf.keras.models.load_model(
            hitnet_file, custom_objects={"hitnet_trafo": hitnet_trafo}
        )
        hitnet.layers[-1].activation = tf.keras.activations.linear

        chargenet = tf.keras.models.load_model(
            chargenet_file, custom_objects={"chargenet_trafo": chargenet_trafo}
        )
        chargenet.layers[-1].activation = tf.keras.activations.linear

        self._batch_size = batch_size

        self._models = (hitnet, chargenet)

        if boundary_guard is not None:
            self._boundary_guard = llh_service.LLHService.init_boundary_guard(
                **boundary_guard
            )
        else:
            self._boundary_guard = None

    @property
    def max_obs_per_batch(self):
        return None

    @property
    def max_hypos_per_batch(self):
        return None

    def request_eval(self, *args, **kwargs):
        """not implemented for CPUClient"""
        self._raise_no_async_error()

    def recv(self, *args, **kwargs):
        """not implemented for CPUClient"""
        self._raise_no_async_error()

    def eval_llh(self, hit_data, evt_data, theta, timeout=None):
        """see llh_client.LLHClient.eval_llh

        timeout is unused for the CPU client but is part of this signature
        to conform to the same interface as the LLHClient"""
        if len(theta.shape) == 1:
            theta = theta[np.newaxis, :]

        batch_size = self._batch_size
        n_hits = len(hit_data)

        hit_table = tf.constant(np.tile(hit_data, (batch_size, 1)), tf.float32)
        evt_table = tf.constant(
            np.repeat(evt_data[np.newaxis, :], batch_size, axis=0), tf.float32
        )
        stop_inds = tf.constant(
            np.arange(n_hits, (batch_size + 1) * n_hits, n_hits), tf.int32
        )

        param_table = np.zeros((batch_size, theta.shape[1]), dtype=np.float32)

        llhs = np.empty(len(theta), dtype=np.float32)
        llh_view = llhs
        while len(theta) > 0:
            n_to_eval = len(theta) if len(theta) < batch_size else batch_size
            param_table[:n_to_eval] = theta[:n_to_eval]
            theta = theta[n_to_eval:]
            param_tensor = tf.constant(param_table)

            batch_llhs = eval_llh.freedom_nllh(
                hit_table,
                evt_table,
                param_tensor,
                stop_inds,
                self._models,
                self._boundary_guard,
            ).numpy()

            llh_view[:n_to_eval] = batch_llhs[:n_to_eval]
            llh_view = llh_view[n_to_eval:]

        assert llh_view.size == 0

        if len(llhs) == 1:
            return llhs[0]
        else:
            return llhs

    @staticmethod
    def _raise_no_async_error():
        raise NotImplementedError(
            "CPUClient does not provide an asynchronous interface"
        )
