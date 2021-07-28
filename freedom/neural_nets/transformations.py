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

    
    def __init__(self, labels, min_energy=0.1, max_energy=1e4, use_pmt_dir=False):
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
        self.use_pmt_dir = use_pmt_dir
        
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
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy, 
                'use_pmt_dir': self.use_pmt_dir}


    def TimeResidual(self, hits, params, Index=1.33):
        '''
        Calculates the difference between the earliest possible arrival time and the measured time of a photon

        Parameters:
        -----------

        hits : tensor
            shape (N, 4+), containing hit DOM position x, y, z and hit time

        params : tensor
            shape (N, len(labels))

        '''
        hitpos = hits[:,:3]
        T_exp = self.CherenkovTime(params, hitpos, Index)
        T_meas = hits[:,3] - params[:, self.time_idx]
        return T_meas - T_exp

    def CherenkovTime(self, params, position, Index=1.33):
        '''
        The earliest possible arrival time of a Cherenkov photon at the given position.
        Assumes neutrino interaction with given params.

        Parameters:
        -----------

        params : tensor
            shape (N, len(labels))
        
        position : tensor
            shape (N, 3), containing hit DOM position x, y, z

        '''
        changle = tf.math.acos(1/Index)
        length = tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy) * 5 #m
        Length = tf.stack([length, length, length], axis=1)

        # closest point on (inf) track, dist, dist along track and direction
        appos, apdist, s, Dir = self.ClosestApproachCalc(params, position)
        a = s - apdist/tf.math.tan(changle) #photon emission distance along track

        return tf.where(a <= 0.,
                        tf.norm(position-params[:,:3], axis=1) * Index/self.speed_of_light,
                        tf.where(a <= length,
                                 (a + apdist/tf.math.sin(changle)*Index) / self.speed_of_light,
                                 (length + tf.norm(position-(params[:,:3] + Length*Dir), axis=1)*Index) / self.speed_of_light
                                )
                       )

    def ClosestApproachCalc(self, params, position):
        '''
        Calculates the closest point to the given position on an infinite track
        with direction specified in given params, the distance of this point and
        the given position as well as the distance of this point along the track.
        Also returns track direction vector of length 1

        Parameters:
        -----------

        params : tensor
            shape (N, len(labels))

        position : tensor
            shape (N, 3), containing hit DOM position x, y, z

        '''
        theta  = params[:, self.zenith_idx]
        phi    = params[:, self.azimuth_idx]
        pos0_x = params[:, self.x_idx]
        pos0_y = params[:, self.y_idx]
        pos0_z = params[:, self.z_idx]

        e_x = -tf.math.sin(theta)*tf.math.cos(phi)
        e_y = -tf.math.sin(theta)*tf.math.sin(phi)
        e_z = -tf.math.cos(theta)

        h_x = position[:,0] - pos0_x
        h_y = position[:,1] - pos0_y
        h_z = position[:,2] - pos0_z

        s = e_x*h_x + e_y*h_y + e_z*h_z

        pos2_x = pos0_x + s*e_x
        pos2_y = pos0_y + s*e_y
        pos2_z = pos0_z + s*e_z

        appos = tf.stack([pos2_x, pos2_y, pos2_z], axis=1)
        apdist = tf.norm(position-appos, axis=1)
        Dir = tf.stack([e_x, e_y, e_z], axis=1)

        return appos, apdist, s, Dir


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
        sinthetadir = tf.sqrt(1 - tf.clip_by_value(tf.math.square(costhetadir), 0, 1)) # can produce NaN on CPU without clip
        # so it is 0 at the poles?
        absdeltaphidir *= sintheta * sinthetadir
        
        dt = hit[:,3] - params[:, self.time_idx]
        ## difference c*t - r
        delta = dt * self.speed_of_light - dist
        tres = self.TimeResidual(hit, params)

        #cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        energy = params[:, self.cascade_energy_idx] + params[:, self.track_energy_idx]
        track_fraction = params[:, self.track_energy_idx] / energy
        
        if self.use_pmt_dir:
            pmt_x = tf.math.sin(hit[:,7]) * tf.math.cos(hit[:,8])
            pmt_y = tf.math.sin(hit[:,7]) * tf.math.sin(hit[:,8])
            pmt_z = tf.math.cos(hit[:,7])
            
            cos_pmtd = tf.clip_by_value((pmt_x*dx + pmt_y*dy + pmt_z*dz)/(dist), -1, 1) # pmt looks to event?
            cos_dird = tf.clip_by_value((dir_x*dx + dir_y*dy + dir_z*dz)/(dist), -1, 1) # event flies to pmt?
            
            #out = [tres, dist, costhetadir, absdeltaphidir, dir_x, dir_y, dir_z, dx, dy, dz, delta, #dt,
            #       hit[:,0], hit[:,1], hit[:,2], hit[:,3], hit[:,5], cos_pmtd, cos_dird, track_fraction]
             
            tres = tf.math.sign(tres)*tf.math.log1p(tf.math.abs(tres))
            dist = tf.math.log1p(dist)
            delta = tf.math.sign(delta)*tf.math.log1p(tf.math.abs(delta))
            dx = tf.math.sign(dx)*tf.math.log1p(tf.math.abs(dx))
            dy = tf.math.sign(dy)*tf.math.log1p(tf.math.abs(dy))
            dz = tf.math.sign(dz)*tf.math.log1p(tf.math.abs(dz))
            
            out = [(tres+10.)/20.,
                   dist/7.3,
                   costhetadir,
                   absdeltaphidir/3.128253,
                   (dir_x+1.)/2.,
                   (dir_y+1.)/2.,
                   (dir_z+1.)/2.,
                   (dx+7.3)/14.3,
                   (dy+7.3)/14.3,
                   (dz+7.3)/14.3,
                   (delta+8.)/16.,
                   (hit[:,0]+5.71e+02)/1.15e+03,
                   (hit[:,1]+5.21e+02)/1.04e+03,
                   (hit[:,2]+5.13e+02)/1.04e+03,
                   (hit[:,3]-5.72e+03)/1.99e+04,
                   hit[:,5],
                   (cos_pmtd+1.)/2.,
                   (cos_dird+1.)/2.,
                   track_fraction,
                  ]
            
        else:
            cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
            track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
            
            out = [delta, dist, costhetadir, absdeltaphidir, dir_x, dir_y, dir_z, dx, dy, dz, hit[:,0], hit[:,1], hit[:,2],
                   hit[:,5], hit[:,6], cascade_energy, track_energy]
        
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
        
        self.det_x = [31.25, 72.37, 41.6, 106.94, 113.19, 57.2, -9.68, -10.97]
        self.det_y = [-72.93, -66.6, 35.49, 27.09, -60.47, -105.52, -79.5, 6.72]
        
    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}
    
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
        
        #cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
        #track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        energy = params[:, self.cascade_energy_idx] + params[:, self.track_energy_idx]
        energy_ratio = params[:, self.track_energy_idx] / params[:, self.cascade_energy_idx]
        track_fraction = params[:, self.track_energy_idx] / energy 
        energy = tf.math.log(tf.clip_by_value(energy, self.min_energy, self.max_energy))
        energy_ratio = tf.clip_by_value(tf.math.log(energy_ratio), -8.7, 8.7)
        
        #x_dists = tf.repeat(params[:, self.x_idx], len(self.det_x)) - tf.tile(self.det_x, [len(params[:, self.x_idx])])
        #y_dists = tf.repeat(params[:, self.y_idx], len(self.det_y)) - tf.tile(self.det_y, [len(params[:, self.y_idx])])
        #string_dist = tf.math.log(1/tf.math.sqrt(x_dists**2 + y_dists**2 + 1))
        #min_dist = tf.math.reduce_max(tf.reshape(string_dist, (len(params[:, self.x_idx]), len(self.det_x))), axis=1)
        #mean_dist = tf.math.reduce_mean(tf.reshape(string_dist, (len(params[:, self.x_idx]), len(self.det_x))), axis=1)
        
        if charge.shape[1] == 2:
            out = tf.stack([
                     charge[:,0]/2.0e4,
                     charge[:,1]/5.41e2,
                     #tf.math.log1p(charge[:,0])/10.0,
                     #tf.math.log1p(charge[:,1])/6.0, #n_channels
                     (params[:, self.x_idx]+750)/1.576e3,
                     (params[:, self.y_idx]+805)/1.577e3,
                     (params[:, self.z_idx]+1115)/1.538e3,
                     (dir_x+1)/2.,
                     (dir_y+1)/2.,
                     (dir_z+1)/2.,
                     #(cascade_energy+2.2)/11.44,
                     #(track_energy+2.3)/11.49,
                     (energy+1.37)/10.6,
                     (energy_ratio+8.7)/17.4,
                     track_fraction,
                     #(min_dist+6.58)/6.571,
                     #(mean_dist+6.68)/2.98,
                    ],
                    axis=1
                    )
        elif charge.shape[1] == 3:
            out = tf.stack([
                     charge[:,0]/2.0e4,
                     charge[:,1]/5.41e2, #n_channels
                     charge[:,2]/44., #n_strings
                     (params[:, self.x_idx]+750)/1.576e3,
                     (params[:, self.y_idx]+805)/1.577e3,
                     (params[:, self.z_idx]+1115)/1.538e3,
                     (dir_x+1)/2.,
                     (dir_y+1)/2.,
                     (dir_z+1)/2.,
                     (energy+1.37)/10.6,
                     (energy_ratio+8.7)/17.4,
                     track_fraction,
                    ],
                    axis=1
                    )
        elif charge.shape[1] == 6:
            cascade_energy = tf.math.log(tf.clip_by_value(params[:, self.cascade_energy_idx], self.min_energy, self.max_energy))
            track_energy = tf.math.log(tf.clip_by_value(params[:, self.track_energy_idx], self.min_energy, self.max_energy))
        
            out = tf.stack([
                     charge[:,0],
                     charge[:,1], #n_channels
                     charge[:,2],
                     charge[:,3], #n_channels
                     charge[:,4],
                     charge[:,5], #n_channels
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
