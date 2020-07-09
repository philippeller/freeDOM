import tensorflow as tf

from freedom.neural_nets.transformations import domnet_trafo

def get_domnet(labels):

    dom_input = tf.keras.Input(shape=(4,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = domnet_trafo(labels=labels)

    h = t(dom_input, params_input)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(512, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    domnet = tf.keras.Model(inputs=[dom_input, params_input], outputs=outputs)

    return domnet
