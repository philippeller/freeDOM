import tensorflow as tf
import tensorflow_addons as tfa

from freedom.neural_nets.transformations import domnet_trafo

def get_domnet(labels, activation="relu"):

    dom_input = tf.keras.Input(shape=(4,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = domnet_trafo(labels=labels)

    s = t(dom_input, params_input)
    #s = tf.keras.layers.BatchNormalization()(s)
    
    h = tf.keras.layers.Dense(32, activation=activation)(s)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(512, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    #h = tf.keras.layers.Dense(1024, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.001)(h)
    #h = tf.keras.layers.Dense(512, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    
    #outputs = tf.keras.layers.concatenate([h, s])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    domnet = tf.keras.Model(inputs=[dom_input, params_input], outputs=outputs)

    return domnet
