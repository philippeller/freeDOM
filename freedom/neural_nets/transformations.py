"""Transformation tensorflow layers"""
import tensorflow as tf
import numpy as np
from scipy import constants


class hitnet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for Hit Net
    Independent angles
    '''
    speed_of_light = constants.c * 1e-9 # c in m / ns

    
    def __init__(self, labels, min_energy=0.1, max_energy=1e4):
        '''
        Parameters:
        -----------

        labels : list
            list of labels corresponding to the data array
        '''
        
        super().__init__()

        self.labels = labels
        self.min_energy = min_energy
        self.max_energy = max_energy
        
        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.time_idx = labels.index('time')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.cascade_energy_idx = labels.index('cascade_energy')
        self.track_energy_idx = labels.index('track_energy')
        
        
        #self.e_cscd_bias = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
        #self.e_cscd_scale = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
        #loc_x_bias
        
        
    def get_config(self):
        return {'labels': self.labels}

    def call(self, hit, params):
        '''
        Parameters:
        -----------

        hit : tensor
            shape (N, 4+), containing hit DOM position x, y, z and hit time

        params : tensor
            shape (N, len(labels))

        '''
        
        cosphi = tf.math.cos(params[:, self.azimuth_idx])
        sinphi = tf.math.sin(params[:, self.azimuth_idx])
        
        sintheta = tf.math.sin(params[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        dx = params[:, self.x_idx] - hit[:,0]
        dy = params[:, self.y_idx] - hit[:,1]
        dz = params[:, self.z_idx] - hit[:,2]
        
        # distance DOM - vertex
        rho = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))     
        
        absdeltaphidir = tf.abs(tf.math.acos(
                                tf.clip_by_value(-tf.math.divide_no_nan((cosphi*dx + sinphi*dy), rho),
                                                 clip_value_min = -1.,
                                                 clip_value_max = +1.,
                                                )
                                            )
                                )

        

        costhetadir = tf.math.divide_no_nan(rho, dist)
        sinthetadir = tf.sqrt(1 - tf.math.square(costhetadir))
        
        # so it is 0 at the poles?
        absdeltaphidir *= sintheta * sinthetadir
        
        dt = hit[:,3] - params[:, self.time_idx]
              
        # difference c*t - r
        delta = dt * self.speed_of_light - dist        

        cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        
        out = tf.stack([
                 delta,
                 dist,
                 costhetadir,
                 absdeltaphidir,
                 dir_x,
                 dir_y,
                 dir_z,
                 dx,
                 dy,
                 dz,
                 hit[:,0],
                 hit[:,1],
                 hit[:,2],
                 hit[:,5],
                 hit[:,6],
                 #hit[:,7],
                 #hit[:,8],
                 cascade_energy,
                 track_energy
                ],
                axis=1
                )    
            
        return out
    
    
class chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for Charget Net
    '''
    
    def __init__(self, labels, min_energy=0.1, max_energy=1e4):
        '''
        Parameters:
        -----------

        labels : list
            list of labels corresponding to the data array
        '''
        
        super().__init__()
        
        self.labels = labels
        self.min_energy = min_energy
        self.max_energy = max_energy
                
        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.cascade_energy_idx = labels.index('cascade_energy')
        self.track_energy_idx = labels.index('track_energy')
        
    def get_config(self):
        return {'labels': self.labels}
    
    def call(self, charge, params):
        '''
        Parameters:
        -----------

        charge : tensor
            shape (N, 1), containing the event total charge

        params : tensor
            shape (N, len(labels))

        '''
        
        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        

        cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))        

        out = tf.stack([
                 charge[:,0],
                 params[:, self.x_idx],
                 params[:, self.y_idx],
                 params[:, self.z_idx],
                 dir_x,
                 dir_y,
                 dir_z,
                 cascade_energy,
                 track_energy,
                ],
                axis=1
                )            

        return out


def test_hitnet_trafo():
    t = hitnet_trafo(labels = ['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'])
    t(np.zeros((1, 9), dtype=np.float32), np.ones((1, 8), dtype=np.float32))
