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
            self.params[:, 3] += time_shifts
        
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

        return (d_X, d_T), d_labels
    
    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        
        return a[p], b[p], c[p]


class charge_trafo_opt(tf.keras.layers.Layer):
    def call(self, charges, theta, dets):
        x_dists = tf.repeat(theta[:,0], len(dets[0])) - tf.tile(dets[0], [len(theta[:,0])])
        y_dists = tf.repeat(theta[:,1], len(dets[1])) - tf.tile(dets[1], [len(theta[:,1])])
        string_dist = 1/(x_dists**2 + y_dists**2 + 0.05**2)
        cos = tf.repeat(tf.math.cos(theta[:,4]), len(dets[0]))
        sin = tf.repeat(tf.math.sin(theta[:,4]), len(dets[0]))
        
        A = (cos*(-x_dists) + sin*(-y_dists)) * tf.math.sqrt(string_dist) + 1
        E_r2 = tf.clip_by_value(A * tf.repeat(theta[:,3], len(dets[0])) * string_dist, 0, 100)
        E_r2 = tf.math.reduce_sum(tf.reshape(E_r2, (len(theta[:,0]), len(dets[0]))), axis=1)
        
        out = tf.stack([
                 charges[:,0]/126.,
                 E_r2/2500.
                ],
                axis=1
                ) 
        return out

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
                 (theta[:,6]-1.)/29.,
                 (theta[:,7])/30.,
                 tf.math.log(theta[:,6]+theta[:,7])/4.1,
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
        rho = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))
        
        cosphi = tf.math.cos(theta[:, 4])
        sinphi = tf.math.sin(theta[:, 4])
        costhetadir = tf.clip_by_value(tf.math.divide_no_nan(rho, dist), 0, 1)
        absdeltaphidir = -tf.math.divide_no_nan((cosphi*dx + sinphi*dy), rho)
        
        dt = theta[:, 3] - hits[:, 0]
        delta = dt - dist
        delta = tf.where(delta<0, -tf.math.log1p(-delta), tf.math.log1p(delta))
        
        dir_x = tf.math.sin(theta[:, 5])*tf.math.cos(theta[:, 4])
        dir_y = tf.math.sin(theta[:, 5])*tf.math.sin(theta[:, 4])
        dir_z = tf.math.cos(theta[:, 5])
        cos_dird = tf.clip_by_value((dir_x*dx + dir_y*dy + dir_z*dz)/(dist), -1, 1)
        
        out = tf.stack([
                 hits[:,0],
                 hits[:,1],
                 hits[:,2],
                 hits[:,3],
                 dx/10.,
                 dy/10.,
                 dz/20.,
                 dir_x,
                 dir_y,
                 dir_z,
                 theta[:,7]/(theta[:,6]+theta[:,7]),
                 #tf.math.log1p(theta[:,7]),
                 tf.math.log1p(dist),
                 tf.math.log1p(rho),
                 delta,
                 tf.math.acos(cos_dird),
                 tf.math.acos(costhetadir),
                 absdeltaphidir/3.128253,
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


def get_hmodel(x_shape, t_shape, trafo, activation='relu'):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    inp = trafo()(x_input, t_input)

    h = tf.keras.layers.Dense(300, activation=activation)(inp)
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
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(300, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model

def get_cmodel(x_shape, t_shape, trafo, activation='relu'):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    inp = trafo()(x_input, t_input)

    h = tf.keras.layers.Dense(150, activation=activation)(inp)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    h = tf.keras.layers.Dense(150, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model
'''
def get_cmodel(x_shape, t_shape, trafo, activation=tfa.activations.mish, dets=None):
    x_input = tf.keras.Input(shape=(x_shape,))
    t_input = tf.keras.Input(shape=(t_shape,))

    inp = trafo()(x_input, t_input, dets=dets)
    
    c, nch, ts = tf.split(inp, [1, 1, 5], 1)

    ls = [ts]
    ls.append(tf.keras.layers.Dense(5, activation=activation)(ts))
    for i in range(50):
        stacked = tf.concat(ls, axis=-1)
        if i == 49:
            ls.append(tf.keras.layers.Dense(100, activation='exponential')(stacked))
        else:
            ls.append(tf.keras.layers.Dense(5, activation=activation)(stacked))
    
    h = tf.keras.layers.Dropout(0.01)(tf.concat(ls, axis=-1))
    h = tf.keras.layers.Dense(100, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.01)(h)
    
    h = tf.concat([h, c], axis=-1)
    h = tf.keras.layers.Dense(30, activation=activation)(h)
    h = tf.concat([h, c], axis=-1)
    h = tf.keras.layers.Dense(30, activation=activation)(h)
    h = tf.concat([h, c], axis=-1)
    h = tf.keras.layers.Dense(30, activation=activation)(h)
    
    h = tf.keras.layers.Dense(30, activation='exponential')(h)
    h = tf.keras.layers.Dense(30, activation=activation)(h)
    #h = tf.concat([h1, h2, c, ts], axis=-1)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    model = tf.keras.Model(inputs=[x_input, t_input], outputs=outputs)
    
    return model
'''

def get_charge_data(events, Truth, nCh=True):
    if nCh:
        x = np.array([list(d) for d in events[:,0]])
    else:
        x = np.array([d[0] for d in events[:,0]]).reshape((len(events),1))
    t = Truth
    return x, t

def get_hit_data(events, Truth):
    x = np.concatenate([d[:,:-1] for d in events[:,1]])
    t = np.repeat(Truth, [len(d) for d in events[:,1]], axis=0)
    return x, t
