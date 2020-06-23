import tensorflow as tf

from freedom.neural_nets.transformations import layernet_trafo

def get_layernet(labels):

    layer_input = tf.keras.Input(shape=(4,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = layernet_trafo(labels=labels)

    h = t(layer_input, params_input)
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

    layernet = tf.keras.Model(inputs=[layer_input, params_input], outputs=outputs)

    return layernet
