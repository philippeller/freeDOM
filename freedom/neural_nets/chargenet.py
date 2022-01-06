import tensorflow as tf
#import tensorflow_addons as tfa
#from tensorflow.keras.layers import Activation

from freedom.neural_nets.transformations import chargenet_trafo

'''
def get_chargenet(labels, n_inp=2, activation='relu', use_nCh=False):

    charge_input = tf.keras.Input(shape=(n_inp,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = chargenet_trafo(labels=labels, use_nCh=use_nCh)

    h = t(charge_input, params_input)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(512, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[charge_input, params_input], outputs=outputs)

    return chargenet

def get_chargenet(labels, n_inp=2, activation='relu', use_nCh=False):

    charge_input = tf.keras.Input(shape=(n_inp,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = chargenet_trafo(labels=labels, use_nCh=use_nCh)
    inp = t(charge_input, params_input)
    
    #c, nch, ts = tf.split(inp, [1, 1, 9], 1)
    #inp = tf.concat([c, ts], axis=-1)

    ls = [inp]
    ls.append(tf.keras.layers.Dense(10, activation=activation)(inp))
    for i in range(50):
        stacked = tf.concat(ls, axis=-1)
        if i == 49:
            ls.append(tf.keras.layers.Dense(100, activation='exponential')(stacked))
        else:
            ls.append(tf.keras.layers.Dense(10, activation=activation)(stacked))
            
    h = tf.keras.layers.Dense(300, activation=activation)(tf.concat(ls, axis=-1))
    h = tf.keras.layers.Dense(200, activation=activation)(h)
    h = tf.keras.layers.Dense(100, activation=activation)(h)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[charge_input, params_input], outputs=outputs)

    return chargenet
'''
def get_chargenet(labels, n_inp=2, activation='relu', final_activation='exponential', use_nCh=False):

    charge_input = tf.keras.Input(shape=(n_inp,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = chargenet_trafo(labels=labels, use_nCh=use_nCh)

    h = t(charge_input, params_input)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=final_activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    
    #h = tf.keras.layers.Dense(1, activation=Activation(tf.math.sinh), trainable=False, kernel_initializer='ones')(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[charge_input, params_input], outputs=outputs)

    return chargenet
