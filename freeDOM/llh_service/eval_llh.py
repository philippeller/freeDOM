from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import tensorflow as tf

# working freeDOM nllh
@tf.function
def freedom_nllh(x, theta, stop_inds, models, charge_ind=4):
    """
    Parameters
     ----------
    x: hits (charge is in column 4)
    theta: hypothesis params
    stop_inds: last index of each separate event in the input
    models: (hitnet, chargenet)
    """

    hitnet = models[0]
    chargenet = models[1]

    # calculate n observations per LLH
    # ensure that sum(n_obs) == len(x) by
    # appending len(x) to the end of stop_inds
    all_inds = tf.concat([[0], stop_inds, [len(x)]], axis=0)
    n_obs = all_inds[1:] - all_inds[:-1]

    # handle zero-padding of dense_theta
    theta = tf.concat([theta, tf.zeros((1, theta.shape[1]), tf.float32)], axis=0)
    dense_theta = tf.repeat(theta, n_obs, axis=0)

    # charge net calculation
    charge_splits = tf.split(x[:, charge_ind], n_obs)
    total_charges = tf.stack([tf.reduce_sum(qs) for qs in charge_splits])[:, tf.newaxis]

    charge_ds = chargenet([total_charges, theta])
    charge_llhs = -tf.math.log(charge_ds / (1 - charge_ds))[:, -1]

    # hit net calculation
    hit_ds = hitnet([x, dense_theta])
    hit_llhs = -tf.math.log(hit_ds / (1 - hit_ds))
    hit_llh_splits = tf.split(hit_llhs, n_obs)
    hit_llh_sums = tf.stack(
        [
            tf.matmul(llh, charge_split[:, tf.newaxis], transpose_a=True)
            for llh, charge_split in zip(hit_llh_splits, charge_splits)
        ]
    )

    # combine hitnet and chargenet
    return hit_llh_sums[:, 0, 0] + charge_llhs


#
# Toy Gaussian functions
#


@tf.function
def eval_llh(x, theta, stop_inds, model):
    # print("tracing eval_llh")
    # tf.print("executing eval_llh")

    # calculate n observations per LLH
    # ensure that sum(n_obs) == len(x) by
    # appending len(x) to the end of stop_inds
    all_inds = tf.concat([[0], stop_inds, [len(x)]], axis=0)
    n_obs = all_inds[1:] - all_inds[:-1]

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
