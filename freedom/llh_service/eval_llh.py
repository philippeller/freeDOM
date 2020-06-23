from __future__ import absolute_import, division, print_function

__author__ = "Aaron Fienberg"

import tensorflow as tf


@tf.function
def freedom_nllh(hit_data, evt_data, theta, stop_inds, models, charge_ind=4):
    """
    hitnet/chargenet llh calculation

    Parameters
     ----------
    hit_data: tf.constant
        table of hit data 
    evt_data: tf.constant
        table of event level data (currently total charge and n hit doms)
    theta: tf.constant
        hypothesis params
    stop_inds: tf.constant
        last index of each separate event in the hit_data table
    models: tuple or list or other indexable object
        (hitnet, chargenet)
    """

    hitnet = models[0]
    chargenet = models[1]

    # calculate n observations per LLH
    # ensure that sum(n_obs) == len(x) by
    # appending len(x) to the end of stop_inds
    all_inds = tf.concat([[0], stop_inds, [len(hit_data)]], axis=0)
    n_obs = all_inds[1:] - all_inds[:-1]

    # handle zero-padding of dense_theta & evt_data
    theta = tf.concat([theta, tf.zeros((1, theta.shape[1]), tf.float32)], axis=0)
    evt_data = tf.concat(
        [evt_data, tf.zeros((1, evt_data.shape[1]), tf.float32)], axis=0
    )
    dense_theta = tf.repeat(theta, n_obs, axis=0)

    # charge net calculation
    charge_llhs = -1 * chargenet([evt_data, theta])[:, -1]

    # hit net calculation
    hit_llhs = -1 * hitnet([hit_data, dense_theta])
    hit_llh_splits = tf.split(hit_llhs, n_obs)
    charge_splits = tf.split(hit_data[:, charge_ind], n_obs)
    hit_llh_sums = tf.stack(
        [
            tf.matmul(llh, charge_split[:, tf.newaxis], transpose_a=True)
            for llh, charge_split in zip(hit_llh_splits, charge_splits)
        ]
    )

    # combine hitnet and chargenet
    return hit_llh_sums[:, 0, 0] + charge_llhs


def wrap_partial_chargenet(partialnet, n_params, n_groups, features_per_group):
    """builds a "flat chargenet" model from a "partial chargenet" model
    
    "flat chargenet" models take a flat array of event level features and output a single
    LLH value per event. This is the format expected in the eval_llh.freedom_nnlh tf.function.

    "partial chargenets" such as layernet and stringnet are fed a number of
    groups of features per event, e.g. per-string features or per-layer features,
    and the outputs for each group within an event must be summed to obtain an event-level LLH value.

    This function wraps "partial chargenets" to create models presenting a "flat chargenet" interface.
    
    Parameters
    ----------
    partialnet : tf.Keras.model
        currently either stringnet or layernet
    n_params : int
        number of hypothesis parameters
    n_groups : int
        number of strings for stringnet, number of layers for layernet
    features_per_group : int
        examples: 5 for stringnet, 4 for layernet

    Returns
    ----------
    tf.Keras.model
        A "chargenet-like" model that takes one fixed-size array of event level features per
        event. This object should be usable as a chargenet in freedom_nllh
    """

    param_inputs = tf.keras.Input(shape=(n_params,))
    params_repeated = tf.repeat(param_inputs, n_groups, axis=0)

    input_layer = tf.keras.Input(shape=(features_per_group * n_groups,))
    reshaped_input = tf.reshape(
        input_layer, (tf.shape(input_layer)[0] * n_groups, features_per_group)
    )

    group_llhs = partialnet([reshaped_input, params_repeated])

    reshaped_llhs = tf.reshape(
        group_llhs, (tf.shape(group_llhs)[0] // n_groups, n_groups)
    )

    sums = tf.reduce_sum(reshaped_llhs, axis=1)

    # match the output shape that would come from chargenet
    reshaped_sums = tf.reshape(sums, (tf.shape(sums)[0], 1))

    return tf.keras.Model(inputs=[input_layer, param_inputs], outputs=reshaped_sums)


#
# Toy Gaussian functions
#


@tf.function
def eval_llh(x, evt_data, theta, stop_inds, model):
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
