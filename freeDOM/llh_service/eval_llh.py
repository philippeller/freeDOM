from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import tensorflow as tf


@tf.function
def eval_llh(x, theta, stop_inds, model):
    #     print("tracing eval_llh")
    #     tf.print("executing eval_llh")

    # calculate n observations per LLH
    # ensure that sum(n_obs) == len(x) by
    # appending len(x) to the end of stop_inds
    stop_inds = tf.concat([stop_inds, [len(x)]], axis=0)
    start_inds = tf.concat([[0], stop_inds[:-1]], axis=0)
    n_obs = stop_inds - start_inds

    # handle zero-padding of dense_theta
    theta = tf.concat([theta, tf.zeros((1, theta.shape[1]), tf.float32)], axis=0)
    dense_theta = tf.repeat(theta, n_obs, axis=0)

    input_tensor = tf.concat([x, dense_theta], axis=1)

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
