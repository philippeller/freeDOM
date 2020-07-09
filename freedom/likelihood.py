import tensorflow as tf
import numpy as np
import pkg_resources

from freedom.neural_nets.transformations import chargenet_trafo, hitnet_trafo, stringnet_trafo, layernet_trafo, domnet_trafo

class LLH():
    def __init__(self,
                 hitnet_file,
                 chargenet_file=None,
                 stringnet_file=None,
                 layernet_file=None,
                 domnet_file=None,
                 chargenet_batchsize=4096,
                 hitnet_batchsize=4096,
                 stringnet_batchsize=4096,
                 layernet_batchsize=4096,
                 domnet_batchsize=4096
                 ):
        '''
        *net_file : str
            location of * model hdf5 file
        *net_batchsize : int
        '''
        
        assert (chargenet_file is None) + (stringnet_file is None) + (layernet_file is None) + (domnet_file is None) == 3, 'Choose either chargenet OR stringnet OR layernet OR domnet'

        self.hitnet = tf.keras.models.load_model(hitnet_file, custom_objects={'hitnet_trafo':hitnet_trafo})
        # set to linear output = logit
        self.hitnet.layers[-1].activation = tf.keras.activations.linear
        self.hitnet.compile()

        if chargenet_file is not None:
            self.chargenet = tf.keras.models.load_model(chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
            self.chargenet.layers[-1].activation = tf.keras.activations.linear
            self.chargenet.compile()
            self.stringnet = None
            self.layernet = None
            self.domnet = None

        elif stringnet_file is not None:
            self.stringnet = tf.keras.models.load_model(stringnet_file, custom_objects={'stringnet_trafo':stringnet_trafo})
            self.stringnet.layers[-1].activation = tf.keras.activations.linear
            self.stringnet.compile()
            self.chargenet = None
            self.layernet = None
            self.domnet = None
            if '_reduced_' in stringnet_file:
                self.allowed_strings = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_strings.npy'))
            else:
                self.allowed_strings = np.arange(86)
            
        elif layernet_file is not None:
            self.layernet = tf.keras.models.load_model(layernet_file, custom_objects={'layernet_trafo':layernet_trafo})
            self.layernet.layers[-1].activation = tf.keras.activations.linear
            self.layernet.compile()
            self.chargenet = None
            self.stringnet = None
            self.domnet = None
            if '_reduced_' in layernet_file:
                self.allowed_layers = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_layers.npy'))
            else:
                self.allowed_layers = 'all'
            
        else:
            self.domnet = tf.keras.models.load_model(domnet_file, custom_objects={'domnet_trafo':domnet_trafo})
            self.domnet.layers[-1].activation = tf.keras.activations.linear
            self.domnet.compile()
            self.chargenet = None
            self.stringnet = None
            self.layernet = None
            if '_reduced_' in domnet_file:
                self.allowed_DOMs = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_DOMs.npy'))
            else:
                self.allowed_DOMs = np.arange(5160)
        
        self.chargenet_batchsize = chargenet_batchsize
        self.hitnet_batchsize = hitnet_batchsize
        self.stringnet_batchsize = stringnet_batchsize
        self.layernet_batchsize = layernet_batchsize
        self.domnet_batchsize = domnet_batchsize
        
    def __call__(self, event, params):
        """Evaluate LLH for a given event + params

        event : dict containing:
            'total_charge' : array
            'hits' : array shape (n_hits, 5)
                each row is (x, y, z) DOM poitions, time, charge, LC flag, ATWD flag
            'strings' : array shape (86, 5)
                each row is x, y, min(z), charge, nChannels
            'layers' : array shape (n_layers, 4)
                each row is nDOMs, z, charge, nChannels
            'doms' : array shape (5160, 4)
                each row is (x, y, z) DOM poitions, charge
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

        # Charge Part
        if self.chargenet is not None:
            inputs = [np.repeat(event['total_charge'][np.newaxis, :], repeats=n_points, axis=0), params]
            charge_llh = -self.chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
        elif self.stringnet is not None:
            strings = event['strings'][self.allowed_strings]
            inputs = []
            inputs.append(np.repeat(strings[:, np.newaxis, :], repeats=n_points, axis=1).reshape(len(strings) * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=len(strings), axis=0).reshape(len(strings) * n_points, -1))

            charge_llhs = -self.stringnet.predict(inputs, batch_size=self.stringnet_batchsize).reshape(len(strings), n_points)
            charge_llh = np.sum(charge_llhs, axis=0)
        elif self.layernet is not None:
            if self.allowed_layers[0] == 'a':
                layers = event['layers']
            else:
                layers = event['layers'][self.allowed_layers]
            inputs = []
            inputs.append(np.repeat(layers[:, np.newaxis, :], repeats=n_points, axis=1).reshape(len(layers) * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=len(layers), axis=0).reshape(len(layers) * n_points, -1))

            charge_llhs = -self.layernet.predict(inputs, batch_size=self.layernet_batchsize).reshape(len(layers), n_points)
            charge_llh = np.sum(charge_llhs, axis=0)
        else:
            doms = event['doms'][self.allowed_DOMs]
            inputs = []
            inputs.append(np.repeat(doms[:, np.newaxis, :], repeats=n_points, axis=1).reshape(len(doms) * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=len(doms), axis=0).reshape(len(doms) * n_points, -1))

            charge_llhs = -self.domnet.predict(inputs, batch_size=self.domnet_batchsize).reshape(len(doms), n_points)
            charge_llh = np.sum(charge_llhs, axis=0)

        # Hit Net
        hits = event['hits']
        n_hits = hits.shape[0]

        if n_hits > 0:
            hit_charges = hits[:, 4]

            inputs = []
            inputs.append(np.repeat(hits[:, np.newaxis, :], repeats=n_points, axis=1).reshape(n_hits * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=n_hits, axis=0).reshape(n_hits * n_points, -1))

            single_hit_llhs = -self.hitnet.predict(inputs, batch_size=self.hitnet_batchsize).reshape(n_hits, n_points)

            all_hits_llh = hit_charges @ single_hit_llhs

        else:
            single_hit_llhs = np.array([])
            all_hits_llh = 0.

        total_llh = all_hits_llh + charge_llh

        return total_llh, charge_llh, all_hits_llh, single_hit_llhs.T
