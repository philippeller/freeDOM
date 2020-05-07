from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import tensorflow as tf


@tf.function
def eval_llh(input_tensor, stop_inds, model):
    # print("tracing eval_llh")
    # tf.print("executing eval_llh")

    # calculate n observations per LLH
    start_inds = tf.concat([[0], stop_inds[:-1]], axis=0)
    n_obs = stop_inds - start_inds

    ds = model(input_tensor)
    llh = tf.math.log(ds / (1 - ds))

    # split llhs by event
    event_splits = tf.split(llh, n_obs)

    llh_sums = tf.stack([tf.reduce_sum(evt) for evt in event_splits])

    return llh_sums


def build_norm_model(model, feature_means, feature_stds):
    """ builds a keras model including input normalization"""
    raw_input = tf.keras.layers.Input(shape=(model.input_shape[0],))

    mean_tensor = tf.constant(feature_means, tf.float32)
    std_tensor = tf.constant(feature_stds, tf.float32)
    normed_input = (raw_input - mean_tensor) / std_tensor

    output = model(normed_input)

    norm_model = tf.keras.Model(inputs=raw_input, outputs=output)

    return norm_model
