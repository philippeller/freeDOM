import tensorflow as tf

def build_q_trafo(detector_xs):
    class q_trafo(tf.keras.layers.Layer):

        def call(self, charges, theta):
            # sum of inverse square of distances to all sensors
            r2s = [tf.math.square(theta[:, 0] - det_x) + tf.math.square(theta[:, 1]) + 0.05**2
                   for det_x in detector_xs]

            inv_sq_sum = tf.reduce_sum(1.0/tf.stack(r2s, axis=-1), axis=-1)

            out = tf.stack([
                     charges[:,0],
                     inv_sq_sum,
                     theta[:,0],
                     theta[:,1],
                     theta[:,2]
                    ],
                    axis=1
                    )
            return out
    
    return q_trafo
    

def build_h_trafo(detector_xs):
    class h_trafo(tf.keras.layers.Layer):
        c = 0.3
        def call(self, hits, theta):
            r2 = tf.math.square(theta[:,0] - hits[:,1]) + tf.math.square(theta[:,1])
            r = tf.math.sqrt(r2)

            delta_t = hits[:,0] - r/self.c 

            out = tf.stack([
                     hits[:,0],
                     hits[:,1],
                     hits[:,2],
                     r,
                     delta_t,
                     theta[:,0],
                     theta[:,1],
                     theta[:,2]
                    ],
                    axis=1
                    )    
            return out
        
    return h_trafo