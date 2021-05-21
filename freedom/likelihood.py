import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pkg_resources

from freedom.neural_nets.transformations import chargenet_trafo, hitnet_trafo, stringnet_trafo, layernet_trafo, domnet_trafo
#from freedom.neural_nets.domnet import combi_activation

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
                 domnet_batchsize=4096,
                 hitnet_trafo=hitnet_trafo,
                 chargenet_trafo=chargenet_trafo
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
            self.domnet = tf.keras.models.load_model(domnet_file, custom_objects={'domnet_trafo':domnet_trafo}) #, 'combi_activation':combi_activation
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
            charge_llhs = -self.chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
            charge_llh = charge_llhs
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

        return total_llh, charge_llh, all_hits_llh, charge_llhs, single_hit_llhs#.T


class upgrade_LLH():
    def __init__(self,
                 DOM_hitnet_file=None,
                 mDOM_hitnet_file=None,
                 DEgg_hitnet_file=None,
                 DOM_chargenet_file=None,
                 mDOM_chargenet_file=None,
                 DEgg_chargenet_file=None,
                 all_chargenet_file=None,
                 chargenet_batchsize=4096,
                 hitnet_batchsize=4096,
                 hitnet_trafo=hitnet_trafo,
                 chargenet_trafo=chargenet_trafo
                 ):
        '''
        *net_file : str
            location of * model hdf5 file
        *net_batchsize : int
        '''

        if DOM_hitnet_file is not None:
            self.DOM_hitnet = tf.keras.models.load_model(DOM_hitnet_file, custom_objects={'hitnet_trafo':hitnet_trafo})
            self.DOM_hitnet.layers[-1].activation = tf.keras.activations.linear
            self.DOM_hitnet.compile()
        else:
            self.DOM_hitnet = None

        if mDOM_hitnet_file is not None:
            self.mDOM_hitnet = tf.keras.models.load_model(mDOM_hitnet_file, custom_objects={'hitnet_trafo':hitnet_trafo})
            self.mDOM_hitnet.layers[-1].activation = tf.keras.activations.linear
            self.mDOM_hitnet.compile()
        else:
            self.mDOM_hitnet = None
            
        if DEgg_hitnet_file is not None:
            self.DEgg_hitnet = tf.keras.models.load_model(DEgg_hitnet_file, custom_objects={'hitnet_trafo':hitnet_trafo})
            self.DEgg_hitnet.layers[-1].activation = tf.keras.activations.linear
            self.DEgg_hitnet.compile()
        else:
            self.DEgg_hitnet = None
            
        if DOM_chargenet_file is not None:
            self.DOM_chargenet = tf.keras.models.load_model(DOM_chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
            self.DOM_chargenet.layers[-1].activation = tf.keras.activations.linear
            self.DOM_chargenet.compile()
        else:
            self.DOM_chargenet = None
            
        if mDOM_chargenet_file is not None:
            self.mDOM_chargenet = tf.keras.models.load_model(mDOM_chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
            self.mDOM_chargenet.layers[-1].activation = tf.keras.activations.linear
            self.mDOM_chargenet.compile()
        else:
            self.mDOM_chargenet = None
            
        if DEgg_chargenet_file is not None:
            self.DEgg_chargenet = tf.keras.models.load_model(DEgg_chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
            self.DEgg_chargenet.layers[-1].activation = tf.keras.activations.linear
            self.DEgg_chargenet.compile()
        else:
            self.DEgg_chargenet = None
            
        if all_chargenet_file is not None:
            self.all_chargenet = tf.keras.models.load_model(all_chargenet_file, custom_objects={'chargenet_trafo':chargenet_trafo})
            self.all_chargenet.layers[-1].activation = tf.keras.activations.linear
            self.all_chargenet.compile()
        else:
            self.all_chargenet = None
        
        self.chargenet_batchsize = chargenet_batchsize
        self.hitnet_batchsize = hitnet_batchsize
        
    def __call__(self, event, params):
        """Evaluate LLH for a given event + params

        event : dict containing:
            'total_charge' : array
            'hits' : array shape (n_hits, 5)
                each row is (x, y, z) DOM poitions, time, charge, LC flag, ATWD flag
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

        # Charge Parts
        if self.DOM_chargenet is not None:
            inputs = [np.repeat(event['total_charge_DOM'][np.newaxis, :], repeats=n_points, axis=0), params]
            charge_llh_DOM = -self.DOM_chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
        else:
            charge_llh_DOM = 0.
        
        if self.mDOM_chargenet is not None:
            inputs = [np.repeat(event['total_charge_mDOM'][np.newaxis, :], repeats=n_points, axis=0), params]
            charge_llh_mDOM = -self.mDOM_chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
        else:
            charge_llh_mDOM = 0.
            
        if self.DEgg_chargenet is not None:
            inputs = [np.repeat(event['total_charge_DEgg'][np.newaxis, :], repeats=n_points, axis=0), params]
            charge_llh_DEgg = -self.DEgg_chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
        else:
            charge_llh_DEgg = 0.
            
        if self.all_chargenet is not None:
            inp = np.stack([event['total_charge_DOM'], event['total_charge_mDOM'], event['total_charge_DEgg']]).reshape(6)
            inputs = [np.repeat(inp[np.newaxis, :], repeats=n_points, axis=0), params]
            charge_llh_all = -self.all_chargenet.predict(inputs, batch_size=self.chargenet_batchsize)[:, 0]
        else:
            charge_llh_all = 0.

        # Hit Nets
        hits = event['hits_DOM']
        n_hits = hits.shape[0]
        if n_hits > 0 and self.DOM_hitnet is not None:
            hit_charges = hits[:, 4]

            inputs = []
            inputs.append(np.repeat(hits[:, np.newaxis, :], repeats=n_points, axis=1).reshape(n_hits * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=n_hits, axis=0).reshape(n_hits * n_points, -1))

            single_hit_llhs_DOM = -self.DOM_hitnet.predict(inputs, batch_size=self.hitnet_batchsize).reshape(n_hits, n_points)

            all_hits_llh_DOM = hit_charges @ single_hit_llhs_DOM

        else:
            single_hit_llhs_DOM = np.array([])
            all_hits_llh_DOM = 0.
            
        hits = event['hits_mDOM']
        n_hits = hits.shape[0]
        if n_hits > 0 and self.mDOM_hitnet is not None:
            hit_charges = hits[:, 4]

            inputs = []
            inputs.append(np.repeat(hits[:, np.newaxis, :], repeats=n_points, axis=1).reshape(n_hits * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=n_hits, axis=0).reshape(n_hits * n_points, -1))

            single_hit_llhs_mDOM = -self.mDOM_hitnet.predict(inputs, batch_size=self.hitnet_batchsize).reshape(n_hits, n_points)

            all_hits_llh_mDOM = hit_charges @ single_hit_llhs_mDOM

        else:
            single_hit_llhs_mDOM = np.array([])
            all_hits_llh_mDOM = 0.
            
        hits = event['hits_DEgg']
        n_hits = hits.shape[0]
        if n_hits > 0 and self.DEgg_hitnet is not None:
            hit_charges = hits[:, 4]

            inputs = []
            inputs.append(np.repeat(hits[:, np.newaxis, :], repeats=n_points, axis=1).reshape(n_hits * n_points, -1))
            inputs.append(np.repeat(params[np.newaxis, :], repeats=n_hits, axis=0).reshape(n_hits * n_points, -1))

            single_hit_llhs_DEgg = -self.DEgg_hitnet.predict(inputs, batch_size=self.hitnet_batchsize).reshape(n_hits, n_points)

            all_hits_llh_DEgg = hit_charges @ single_hit_llhs_DEgg

        else:
            single_hit_llhs_DEgg = np.array([])
            all_hits_llh_DEgg = 0.

        
        total_llh_DOM = all_hits_llh_DOM + charge_llh_DOM
        total_llh_mDOM = all_hits_llh_mDOM + charge_llh_mDOM
        total_llh_DEgg = all_hits_llh_DEgg + charge_llh_DEgg
        total_llh = total_llh_DOM + total_llh_mDOM + total_llh_DEgg #+ charge_llh_all

        return total_llh, total_llh_DOM, total_llh_mDOM, total_llh_DEgg #, charge_llh_all
