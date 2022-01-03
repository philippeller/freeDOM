"""Transformation tensorflow layers"""
import tensorflow as tf
#import numpy as np
from scipy import constants
import pkg_resources


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


    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}


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
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))     

        dt = hit[:, 3] - params[:, self.time_idx]
        delta = dt * self.speed_of_light - dist

        cascade_energy = tf.math.log1p(params[:, self.cascade_energy_idx])
        track_energy = tf.math.log1p(params[:, self.track_energy_idx])
        
        pmt_x = tf.math.sin(hit[:,7]) * tf.math.cos(hit[:,8])
        pmt_y = tf.math.sin(hit[:,7]) * tf.math.sin(hit[:,8])
        pmt_z = tf.math.cos(hit[:,7])
        
        dx = dx/dist
        dy = dy/dist
        dz = dz/dist

        dist = tf.math.log1p(dist)
        delta = tf.where(delta<0, -tf.math.log1p(-delta), tf.math.log1p(delta))

        out = [dist/7.5,
               dir_x,
               dir_y,
               dir_z,
               dx,
               dy,
               dz,
               delta/8.0,
               cascade_energy/9.0,
               track_energy/10.0,
               pmt_x,
               pmt_y,
               pmt_z,
               (hit[:,0]+5.71e+02)/1.15e+03,
               (hit[:,1]+5.21e+02)/1.04e+03,
               (hit[:,2]+5.13e+02)/1.04e+03,
               (hit[:,3]-5.72e+03)/1.99e+04,
               hit[:,5]
              ]
        
        out = tf.stack(out, axis=1)

        return out


class domnet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for domnet
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
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, dom, params):
        '''
        Parameters:
        -----------

        dom : tensor
            shape (N, 4), containing hit dom position x, y, z, and charge

        params : tensor
            shape (N, len(labels))

        '''
        
        cosphi = tf.math.cos(params[:, self.azimuth_idx])
        sinphi = tf.math.sin(params[:, self.azimuth_idx])
        sintheta = tf.math.sin(params[:, self.zenith_idx])
        
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        dx = params[:, self.x_idx] - dom[:,0]
        dy = params[:, self.y_idx] - dom[:,1]
        dz = params[:, self.z_idx] - dom[:,2]
        
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
        sinthetadir = tf.sqrt(1 - tf.clip_by_value(tf.math.square(costhetadir), 0, 1)) # can produce NaN on CPU without clip
        
        # so it is 0 at the poles?
        absdeltaphidir *= sintheta * sinthetadir

        cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))

        out = tf.stack([
                 dom[:,0],
                 dom[:,1],
                 dom[:,2],
                 dom[:,3],
                 dist,
                 costhetadir,
                 absdeltaphidir,
                 dir_x,
                 dir_y,
                 dir_z,
                 dx,
                 dy,
                 dz,
                 cascade_energy,
                 track_energy
                ],
                axis=1
                )    
            
        return out
    
class stringnet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for stringnet
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
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, string, params):
        '''
        Parameters:
        -----------

        string : tensor
            shape (N, 5), containing hit string position x, y, min(z), charge, and nChannels

        params : tensor
            shape (N, len(labels))

        '''
        
        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        dx = params[:, self.x_idx] - string[:,0]
        dy = params[:, self.y_idx] - string[:,1]
        dz = params[:, self.z_idx] - string[:,2]
        
        # distance string - vertex
        rho = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))

        cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        
        out = tf.stack([
                 string[:,0],
                 string[:,1],
                 string[:,2],
                 string[:,3],
                 string[:,4],
                 rho,
                 dir_x,
                 dir_y,
                 dir_z,
                 dx,
                 dy,
                 dz,
                 cascade_energy,
                 track_energy
                ],
                axis=1
                )    
            
        return out
    
class layernet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for layernet
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
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, layer, params):
        '''
        Parameters:
        -----------

        layer : tensor
            shape (N, 4), containing hit layer nDOMs, z position, charge, and nChannels

        params : tensor
            shape (N, len(labels))

        '''
        
        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        dz = params[:, self.z_idx] - layer[:,1]
        
        cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        
        out = tf.stack([
                 layer[:,0],
                 layer[:,1],
                 layer[:,2],
                 layer[:,3],
                 dir_x,
                 dir_y,
                 dir_z,
                 params[:, self.x_idx],
                 params[:, self.y_idx],
                 dz,
                 cascade_energy,
                 track_energy
                ],
                axis=1
                )    
            
        return out

class chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for Charget Net
    '''
    
    def __init__(self, labels, min_energy=0.1, max_energy=1e4, use_nCh=False):
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
        self.use_nCh = use_nCh
                
        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.cascade_energy_idx = labels.index('cascade_energy')
        self.track_energy_idx = labels.index('track_energy')
        
    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy, 'use_nCh': self.use_nCh}
    
    def call(self, charge, params):
        '''
        Parameters:
        -----------

        charge : tensor
            shape (N, 2), containing the event total charge and number of hit DOMs

        params : tensor
            shape (N, len(labels))

        '''
        
        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        cascade_energy = tf.math.log1p(params[:, self.cascade_energy_idx])
        track_energy = tf.math.log1p(params[:, self.track_energy_idx])

        if self.use_nCh:
            out = tf.stack([
                     tf.math.log1p(charge[:,0])/10.0,
                     tf.math.log1p(charge[:,1])/6.0, #n_channels
                     (params[:, self.x_idx])/8.0e2,
                     (params[:, self.y_idx])/8.0e2,
                     (params[:, self.z_idx]+350)/7.5e2,
                     dir_x,
                     dir_y,
                     dir_z,
                     cascade_energy/9.0,
                     track_energy/10.0,
                    ],
                    axis=1
                    )
        else:
            out = tf.stack([
                     tf.math.log1p(charge[:,0])/10.0,
                     (params[:, self.x_idx])/8.0e2,
                     (params[:, self.y_idx])/8.0e2,
                     (params[:, self.z_idx]+350)/7.5e2,
                     dir_x,
                     dir_y,
                     dir_z,
                     cascade_energy/9.0,
                     track_energy/10.0,
                    ],
                    axis=1
                    )

        return out
    
class c_trafo(tf.keras.layers.Layer):
    def call(self, theta):
        dir_x = tf.math.sin(theta[:, 5]) * tf.math.cos(theta[:, 4])
        dir_y = tf.math.sin(theta[:, 5]) * tf.math.sin(theta[:, 4])
        dir_z = tf.math.cos(theta[:, 5])
        cascade_energy = tf.math.log(tf.clip_by_value(theta[:,6], 0.1, 1e4))
        track_energy = tf.math.log(tf.clip_by_value(theta[:,7], 0.1, 1e4))
        
        out = tf.stack([
                 (theta[:,0]+750)/1.576e3,
                 (theta[:,1]+805)/1.577e3,
                 (theta[:,2]+1115)/1.538e3,
                 (dir_x+1)/2.,
                 (dir_y+1)/2.,
                 (dir_z+1)/2.,
                 (cascade_energy+2.2)/11.44,
                 (track_energy+2.3)/11.49,
                ],
                axis=1
                )    
        return out

class prior_trafo(tf.keras.layers.Layer):
    def __init__(self, trainable=False, name='InputTransformer', **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        
    def call(self, inputs, **kwargs):
        zeni = (tf.math.cos(inputs[:,5]) + self.minima[5]) / (self.maxima[5] + abs(self.minima[5]))
        cascade = (tf.math.log(inputs[:, 6]) + self.minima[6]) / (self.maxima[6] + abs(self.minima[6]))
        track = (tf.math.log(tf.clip_by_value(inputs[:, 7], clip_value_min=1e-3, clip_value_max=1e12)) + abs(self.minima[7])) / (self.maxima[7] + abs(self.minima[7]))
        
        out = tf.stack([
                 (inputs[:,0] + abs(self.minima[0])) / (self.maxima[0] + abs(self.minima[0])),
                 (inputs[:,1] + abs(self.minima[1])) / (self.maxima[1] + abs(self.minima[1])),
                 (inputs[:,2] + abs(self.minima[2])) / (self.maxima[2] + abs(self.minima[2])),
                 (inputs[:,4] + abs(self.minima[4])) / (self.maxima[4] + abs(self.minima[4])),
                 zeni,
                 cascade,
                 track,
                ],
                axis=1
                )    
        return out
    
    @property
    def maxima(self) -> tf.constant:
        a = tf.constant(
            [8.26402465820312500000 * 1e2, 7.72468811035156250000 * 1e2, 4.23771453857421875000 * 1e2,0, 2 * constants.pi, 1,
             9.20983600616455078125, 9.19035434722900390625])
        return a
    
    @property
    def minima(self) -> tf.constant:
        a = tf.constant([-750.53778076171875, -805.071533203125, -1114.884033203125,0, 0, -1, -2.23472332954406738281,
                         -6.90775537490844726562])
        return a


def test_hitnet_trafo():
    t = hitnet_trafo(labels = ['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'])
    t(tf.constant([[0.]*9]), tf.constant([[1.]*8]))
