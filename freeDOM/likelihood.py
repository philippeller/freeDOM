import tensorflow as tf
import numpy as np

from transformations import chargenet_trafo, hitnet_trafo



class LLH():
    def __init__(self,
                 hitnet_file,
                 chargenet_file,
                 epsilon = 1e-10,
                 chargenet_batchsize = 4096,
                 hitnet_batchsize = 2**15,
                 ):
        '''
        epsilon : float
            value to clip discriminator output at to avoid division by zero
        '''
        self.hitnet = tf.keras.models.load_model(hitnet_file, custom_objects={'hitnet_trafo':hitnet_trafo})
        self.chargenet = tf.keras.models.load_model(chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
        
        self.epsilon = epsilon
        self.chargenet_batchsize = chargenet_batchsize
        self.hitnet_batchsize = hitnet_batchsize
        
    def __call__(self, event, params):
        """Evaluate LLH for a given event + params

        event : dict containing:
            'total_charge' : float
            'hits' : array shape (n_hits, 5)
                each row is (x, y, z) DOM poitions, time, charge
        params : ndarray
            shape (n_likelihood_points, len(labels)) 

        
        Returns:
        --------
        total_llh : ndarray
        charge_llh : ndarray
        all_hits_llh : ndarray
        single_hit_llhs : ndarray

        """

        if params.ndim == 1:
            params = np.array([params])
        n_points = params.shape[0]    

        # Charge Net
        inputs = [np.repeat(event['total_charge'], repeats=n_points)[:, np.newaxis], params]
        score = self.chargenet.predict(inputs, batch_size=self.chargenet_batchsize)
        score = score[:, 0]
        score = np.clip(score, self.epsilon, 1-self.epsilon)
        charge_llh = -(np.log(score) - np.log(1 - score))

        # Hit Net
        hits = event['hits']
        n_hits = hits.shape[0]

        if n_hits > 0:
            hit_charges = hits[:, 4]

            inputs = []
            inputs.append(np.repeat(hits[:, np.newaxis, :], repeats=n_points, axis=1).reshape(n_hits * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=n_hits, axis=0).reshape(n_hits * n_points, -1))

            score = self.hitnet.predict(inputs, batch_size=self.hitnet_batchsize)
            score = score.reshape(n_hits, n_points)
            score = np.clip(score, self.epsilon, 1-self.epsilon)
            single_hit_llhs = -(np.log(score) - np.log(1 - score))

            all_hits_llh = hit_charges @ single_hit_llhs

        else:
            single_hit_llhs = np.array([])
            all_hits_llh = 0.

        total_llh = all_hits_llh + charge_llh

        return total_llh, charge_llh, all_hits_llh, single_hit_llhs.T
