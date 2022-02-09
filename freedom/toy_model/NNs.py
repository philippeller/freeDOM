import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=4096, shuffle='free'):
        assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."
        
        self.batch_size = int(batch_size/2) # half true labels half false labels
        self.data = x
        self.params = t
        
        #spread absolute time values
        if len(self.data[0]) > 2:
            time_shifts = np.random.normal(0, 5, len(self.data))
            self.data[:, 0] += time_shifts
            self.params[:, 3] += time_shifts
            
        if shuffle == 'inDOM':
            self.shuffle_params_inDOM()
        else:
            self.shuffled_params = []
        
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
        #if len(self.shuffled_params) > 0:
            #self.shuffle_params_inDOM()
        
    def shuffle_params_inDOM(self, dom_idx=5):
        shuffled_params = np.empty_like(self.params)
        u,c = np.unique(self.data[:, dom_idx], return_counts=True)
        for DOM_index in u[c>1]:
            mask = self.data[:, dom_idx] == DOM_index
            shuffled_params[mask] = np.random.permutation(self.params[mask])
        
        self.shuffled_params = shuffled_params

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        x = np.take(self.data, indexes_temp, axis=0)
        t = np.take(self.params, indexes_temp, axis=0)
        if len(self.shuffled_params) == 0:
            tr = np.random.permutation(t)
        else:
            tr = np.take(self.shuffled_params, indexes_temp, axis=0)

        d_true_labels = np.ones((self.batch_size, 1), dtype=x.dtype)
        d_false_labels = np.zeros((self.batch_size, 1), dtype=x.dtype)
        
        d_X = np.append(x, x, axis=0)
        d_T = np.append(t, tr, axis=0)
        d_labels = np.append(d_true_labels, d_false_labels)
        
        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return (d_X, d_T), d_labels
    
    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        
        return a[p], b[p], c[p]


class charge_trafo(tf.keras.layers.Layer):
    def call(self, charges, theta, dets=None):
        out = tf.stack([
                 charges[:,0]/126.,
                 charges[:,1]/18.,
                 (theta[:,0]+12.)/24.,
                 (theta[:,1]+12.)/24.,
                 (theta[:,3]+3.)/37.,
                 (tf.math.sin(theta[:,4])+1.)/2.,
                 (tf.math.cos(theta[:,4])+1.)/2.,
                ],
                axis=1
                ) 
        return out    

class charge_trafo_3D(tf.keras.layers.Layer):
    def call(self, charges, theta, dets=None):
        out = tf.stack([
                 tf.math.log1p(charges[:,0])/6.7,
                 tf.math.log1p(charges[:,1])/4.6,
                 (theta[:,0]+2.)/14.,
                 (theta[:,1]+12.)/16.,
                 (theta[:,2]+18.)/36.,
                 (tf.math.sin(theta[:,5])*tf.math.cos(theta[:,4])+1.)/2.,
                 (tf.math.sin(theta[:,5])*tf.math.sin(theta[:,4])+1.)/2.,
                 (tf.math.cos(theta[:,5])+1.)/2.,
                 (theta[:,6]-1.)/19.,
                 (theta[:,7])/20.
                ],
                axis=1
                ) 
        return out

class hit_trafo(tf.keras.layers.Layer):
    def call(self, hits, theta, dets=None):
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

class hit_trafo_3D(tf.keras.layers.Layer):
    def call(self, hits, theta, dets=None):
        
        dx = theta[:, 0] - hits[:, 1]
        dy = theta[:, 1] - hits[:, 2]
        dz = theta[:, 2] - hits[:, 3]
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))
        
        dt = hits[:, 0] - theta[:, 3]
        delta = dt - dist
        delta = tf.where(delta<0, -tf.math.log1p(-delta), tf.math.log1p(delta))
        
        dir_x = tf.math.sin(theta[:, 5])*tf.math.cos(theta[:, 4])
        dir_y = tf.math.sin(theta[:, 5])*tf.math.sin(theta[:, 4])
        dir_z = tf.math.cos(theta[:, 5])
        
        out = tf.stack([
                 hits[:,0],
                 hits[:,1],
                 hits[:,2],
                 hits[:,3],
                 tf.math.log1p(dist),
                 dx/dist,
                 dy/dist,
                 dz/dist,
                 delta,
                 dir_x,
                 dir_y,
                 dir_z,
                 theta[:,6],
                 theta[:,7],
                ],
                axis=1
                )    
        return out
    
class dom_trafo_3D(tf.keras.layers.Layer):
    def call(self, doms, theta, dets=None):
        
        dx = theta[:, 0] - doms[:, 1]
        dy = theta[:, 1] - doms[:, 2]
        dz = theta[:, 2] - doms[:, 3]
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))
        
        dir_x = tf.math.sin(theta[:, 5])*tf.math.cos(theta[:, 4])
        dir_y = tf.math.sin(theta[:, 5])*tf.math.sin(theta[:, 4])
        dir_z = tf.math.cos(theta[:, 5])
        
        out = tf.stack([
                 doms[:,0],
                 doms[:,1],
                 doms[:,2],
                 doms[:,3],
                 tf.math.log1p(dist),
                 dx/dist,
                 dy/dist,
                 dz/dist,
                 dir_x,
                 dir_y,
                 dir_z,
                 theta[:,6],
                 theta[:,7],
                ],
                axis=1
                )    
        return out


def get_hmodel(x_shape, t_shape, trafo, activation='relu', final_activation='exponential'):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    inp = trafo()(x_input, t_input)

    h = tf.keras.layers.Dense(250, activation=activation)(inp)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(250, activation=final_activation)(h)
    #h = tf.keras.layers.Dropout(0.01)(h)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model

def get_cmodel(x_shape, t_shape, trafo, activation='relu', final_activation='exponential'):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    inp = trafo()(x_input, t_input)

    h = tf.keras.layers.Dense(150, activation=activation)(inp)
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

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model


def get_hit_data(events, Truth):
    x = np.concatenate([d for d in events[:, 1]])
    t = np.repeat(Truth, [len(d) for d in events[:, 1]], axis=0)
    return x, t

def get_charge_data(events, Truth, nCh=True):
    if nCh:
        x = np.array([list(d) for d in events[:, 0]])
    else:
        x = np.array([d[0] for d in events[:, 0]]).reshape((len(events),1))
    t = Truth
    return x, t

def get_dom_data(events, Truth):
    x = np.concatenate([d for d in events[:, 0]])
    t = np.repeat(Truth, [len(events[0,0])]*len(events), axis=0)
    return x, t 
