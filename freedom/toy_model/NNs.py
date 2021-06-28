import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=4096):
        
        self.batch_size = int(batch_size/2) # half true labels half false labels
        self.data = x
        self.params = t
        
        #spread absolute time values
        if len(self.data[0]) > 2:
            time_shifts = np.random.normal(0, 5, len(self.data))
            self.data[:, 0] += time_shifts
            self.params[:, 2] += time_shifts
        
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes) # mix between batches

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        x = np.take(self.data, indexes_temp, axis=0)
        t = np.take(self.params, indexes_temp, axis=0)

        d_true_labels = np.ones((self.batch_size, 1), dtype=x.dtype)
        d_false_labels = np.zeros((self.batch_size, 1), dtype=x.dtype)
        
        d_X = np.append(x, x, axis=0)
        d_T = np.append(t, np.random.permutation(t), axis=0)
        d_labels = np.append(d_true_labels, d_false_labels)
        
        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return [d_X, d_T], d_labels
    
    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        
        return a[p], b[p], c[p]


class charge_trafo(tf.keras.layers.Layer):
    def call(self, charges, theta, dets): #
        #x_dists = tf.repeat(theta[:,0], len(dets[0])) - tf.tile(dets[0], [len(theta[:,0])])
        #y_dists = tf.repeat(theta[:,1], len(dets[1])) - tf.tile(dets[1], [len(theta[:,1])])
        #string_dist = 1/(x_dists**2 + y_dists**2 + 0.05**2)
        #cos = tf.repeat(tf.math.cos(theta[:,4]), len(dets[0]))
        #sin = tf.repeat(tf.math.sin(theta[:,4]), len(dets[0]))
        
        #A = (cos*(-x_dists) + sin*(-y_dists)) * tf.math.sqrt(string_dist) + 1
        #E_r2 = tf.clip_by_value(A * tf.repeat(theta[:,3], len(dets[0])) * string_dist, 0, 100)
        #E_r2 = tf.math.reduce_sum(tf.reshape(E_r2, (len(theta[:,0]), len(dets[0]))), axis=1)
        
        out = tf.stack([
                 charges[:,0]/126.,
                 charges[:,1]/18.,
                 (theta[:,0]+12.)/24.,
                 (theta[:,1]+12.)/24.,
                 (theta[:,3]+3.)/37.,
                 (tf.math.sin(theta[:,4])+1.)/2.,
                 (tf.math.cos(theta[:,4])+1.)/2.,
                 #E_r2/2500.
                ],
                axis=1
                ) 
        return out
'''
class charge_trafo(tf.keras.layers.Layer):
    def call(self, charges, theta, dets): #
        x_dists = tf.repeat(theta[:,0], len(dets[0])) - tf.tile(dets[0], [len(theta[:,0])])
        y_dists = tf.repeat(theta[:,1], len(dets[1])) - tf.tile(dets[1], [len(theta[:,1])])
        string_dist = tf.reshape(tf.math.sqrt(x_dists**2 + y_dists**2), (len(theta[:,0]), len(dets[0]))) # + 0.05**2
        
        d0 = string_dist[:,0]
        d1 = string_dist[:,1]
        d2 = string_dist[:,2]
        d3 = string_dist[:,3]
        d4 = string_dist[:,4]
        d5 = string_dist[:,5]
        d6 = string_dist[:,6]
        d7 = string_dist[:,7]
        d8 = string_dist[:,8]
        d9 = string_dist[:,9]
        d10 = string_dist[:,10]
        d11 = string_dist[:,11]
        d12 = string_dist[:,12]
        d13 = string_dist[:,13]
        d14 = string_dist[:,14]
        d15 = string_dist[:,15]
        d16 = string_dist[:,16]
        d17 = string_dist[:,17]
        d18 = string_dist[:,18]
        d19 = string_dist[:,19]
        d20 = string_dist[:,20]
        d21 = string_dist[:,21]
        d22 = string_dist[:,22]
        d23 = string_dist[:,23]
        d24 = string_dist[:,24]
        
        out = tf.stack([
                 charges[:,0]/126.,
                 (theta[:,3]+3.)/37.,
                 (tf.math.sin(theta[:,4])+1.)/2.,
                 (tf.math.cos(theta[:,4])+1.)/2.,
                 d0/31.,
                 d1/28.,
                 d2/25.,
                 d3/28.,
                 d4/31.,
                 d5/28.,
                 d6/24.,
                 d7/21.,
                 d8/24.,
                 d9/28.,
                 d10/25.,
                 d11/21.,
                 d12/17.,
                 d13/21.,
                 d14/25.,
                 d15/28.,
                 d16/24.,
                 d17/21.,
                 d18/24.,
                 d19/28.,
                 d20/31.,
                 d21/28.,
                 d22/25.,
                 d23/28.,
                 d24/31.,
                ],
                axis=1
                ) 
        return out
'''
class hit_trafo(tf.keras.layers.Layer):
    def call(self, hits, theta):
        dx = theta[:,0] - hits[:,1]
        dy = theta[:,1] - hits[:,2]
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))
        dt = theta[:,2] - hits[:,0]
        
        out = tf.stack([
                 hits[:,0],
                 hits[:,1],
                 hits[:,2],
                 dx, #theta[:,0],
                 dy, #theta[:,1],
                 dt, #theta[:,2],
                 tf.math.sin(theta[:,4]),
                 tf.math.cos(theta[:,4]),
                 dist,
                ],
                axis=1
                )    
        return out
    
def my_layer(h): # not used so far
    h1 = tf.keras.layers.Dense(60, activation='relu')(h)
    h2 = tf.keras.layers.Dense(4, activation='exponential')(h)
    h = tf.keras.layers.Concatenate()([h1,h2])
    h = tf.keras.layers.Dropout(0.001)(h)
    return h
        
def get_model(x_shape, t_shape, trafo, activation='relu', dets=None):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    if np.all(dets) == None:
        inp = trafo()(x_input, t_input)
    else:
        inp = trafo()(x_input, t_input, dets=dets)

    h = tf.keras.layers.Dense(16, activation=activation)(inp)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(inp)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(16, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model

def get_charge_data(events, Truth, nCh=True):
    if nCh:
        x = np.array([list(d) for d in events[:,0]])
    else:
        x = np.array([d[0] for d in events[:,0]]).reshape((len(events),1))
    t = Truth
    return x, t

def get_hit_data(events, Truth):
    x = np.concatenate([d[:,:3] for d in events[:,1]])
    t = np.repeat(Truth, [len(d) for d in events[:,1]], axis=0)
    return x, t
