import tensorflow as tf
#import tensorflow_addons as tfa

from freedom.neural_nets.transformations import hitnet_trafo
'''
def get_hitnet(labels, activation="relu"):

    hits_input = tf.keras.Input(shape=(10,)) #8
    params_input = tf.keras.Input(shape=(len(labels),))
    
    t = hitnet_trafo(labels=labels)

    s = t(hits_input, params_input)
    #s = tf.keras.layers.BatchNormalization()(s)
    
    h = tf.keras.layers.Dense(32, activation=activation)(s)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(512, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(1024, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(512, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    
    #h = tf.keras.layers.concatenate([h, s])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hits_input, params_input], outputs=outputs)

    return hitnet
'''
def get_hitnet(labels, activation="relu", final_activation='exponential'):

    hits_input = tf.keras.Input(shape=(10,))
    params_input = tf.keras.Input(shape=(len(labels),))
    
    t = hitnet_trafo(labels=labels)

    s = t(hits_input, params_input)
    
    h = tf.keras.layers.Dense(300, activation=activation)(s) #1000
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(300, activation=final_activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    
    #h = tf.keras.layers.Dense(1, activation=tf.keras.layers.Activation(tf.math.sinh), trainable=False, kernel_initializer='ones')(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hits_input, params_input], outputs=outputs)

    return hitnet
'''
def get_hitnet(labels, activation="relu", final_activation=tfa.activations.mish):

    hits_input = tf.keras.Input(shape=(10,))
    params_input = tf.keras.Input(shape=(len(labels),))
    
    t = hitnet_trafo(labels=labels)

    s = t(hits_input, params_input)
    sp, sx = tf.split(s, [10, 8], 1)
    
    h = tf.keras.layers.Dense(300, activation=activation)(sp)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    
    h = tf.keras.layers.Concatenate()([h, sx])
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=final_activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hits_input, params_input], outputs=outputs)

    return hitnet
'''