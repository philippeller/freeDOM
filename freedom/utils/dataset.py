"""Module to create tf.data.DataSet and DataGenerator instances for training"""
import os
import pkg_resources
import numpy as np
import tensorflow as tf
from freedom.utils.i3cols_dataloader import load_hits, load_charges, load_strings, load_layers, load_doms, get_energies, get_params
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Data():
    def __init__(self,
                 dirs=['/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols'],
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                ):
        

        self.dirs = dirs
        self.labels = labels
        self.geo = geo

    def get_hitnet_data(self, train_batch_size=1024, test_batch_size=256, shuffle_block_size=2**14, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_hits(dir=dir, labels=self.labels, geo=self.geo))
        single_hits = np.concatenate([d[0] for d in data])
        repeated_params = np.concatenate([d[1] for d in data])
        
        hits_train, hits_test, params_train, params_test = train_test_split(single_hits,
                                                                            repeated_params,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        
        train = self.get_dataset(hits_train, params_train, batch_size=train_batch_size, shuffle_block_size=shuffle_block_size)
        test = self.get_dataset(hits_test, params_test, batch_size=test_batch_size, test=True, shuffle_block_size=shuffle_block_size)
        
        return train, test
        
        
    def get_chargenet_data(self, train_batch_size=1024, test_batch_size=256, shuffle_block_size=2**14, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_charges(dir=dir, labels=self.labels))
        total_charge = np.concatenate([d[0] for d in data])
        params = np.concatenate([d[1] for d in data])

        charge_train, charge_test, params_train, params_test = train_test_split(total_charge,
                                                                                params, 
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        
        train = self.get_dataset(charge_train, params_train, batch_size=train_batch_size, shuffle_block_size=shuffle_block_size)
        test = self.get_dataset(charge_test, params_test, batch_size=test_batch_size, test=True, shuffle_block_size=shuffle_block_size)
        
        return train, test
        
    def get_stringnet_data(self, train_batch_size=1024, test_batch_size=256, shuffle_block_size=2**14, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_strings(dir=dir, labels=self.labels))
        string_charges = np.concatenate([d[0] for d in data])
        params = np.concatenate([d[1] for d in data])

        string_train, string_test, params_train, params_test = train_test_split(string_charges,
                                                                                params, 
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        
        train = self.get_dataset(string_train, params_train, batch_size=train_batch_size, shuffle_block_size=shuffle_block_size)
        test = self.get_dataset(string_test, params_test, batch_size=test_batch_size, test=True, shuffle_block_size=shuffle_block_size)
        
        return train, test
        
        
    def get_dataset(self, x, t, shuffle_block_size=2**14, batch_size=1024, test=False):
        '''
        get a tensorflow dataset for likelihood approximation

        
        Parameters:
        -----------
        x : ndarray
            observations
        t : ndarray
            parameters        
        shuffle_block_size : int
            block size over which to shuffle, should be multiple of batch_size
        batch_size : int
        test : bool
            no shuffling, prefetching and caching
        
        Returns:
        --------
        
        tf.data.Dataset
            with structure ((x, t), y) for training
        
        '''
        
        N = x.shape[0]
        assert t.shape[0] == N
        
        d_x = tf.data.Dataset.from_tensor_slices(x)
        d_t = tf.data.Dataset.from_tensor_slices(t)

        d_true_labels = tf.data.Dataset.from_tensor_slices(np.ones((N, 1), dtype=x.dtype))
        d_false_labels = tf.data.Dataset.from_tensor_slices(np.zeros((N, 1), dtype=x.dtype))

        d_xs = tf.data.Dataset.from_tensor_slices([d_x, d_x]).interleave(lambda x : x)
        d_ts = tf.data.Dataset.from_tensor_slices([d_t, d_t.shuffle(shuffle_block_size)]).interleave(lambda x : x)
        d_ys = tf.data.Dataset.from_tensor_slices([d_true_labels, d_false_labels]).interleave(lambda x : x)
        
        
        dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((d_xs, d_ts)), d_ys))
  
        return dataset.batch(batch_size)
        
        # batch before shuffle
        
#         d_x_windows = d_x.window(shuffle_block_size)
#         d_t_windows = d_t.window(shuffle_block_size)
#         d_true_labels_windows = d_true_labels.window(shuffle_block_size)
#         d_false_labels_windows = d_false_labels.window(shuffle_block_size)
        
        
#         # shuffled, i.e. p(x)p(t) vs p(x,t)
        
#         d_X = tf.data.Dataset.concatenate(d_x_windows, d_x_windows)
#         d_T = tf.data.Dataset.concatenate(d_t_windows, d_t_windows.map(lambda x: x.shuffle(shuffle_block_size)))
        
#         d_inputs = tf.data.Dataset.zip((d_X, d_T))        
#         d_outputs = tf.data.Dataset.concatenate(d_true_labels_windows, d_false_labels_windows)


#         dataset = tf.data.Dataset.zip((d_inputs, d_outputs))
        
#         if test:
#             return dataset.batch(batch_size)
        
        
#         batched = dataset.shuffle(2*N).batch(batch_size=batch_size)
        
#         prefetched = batched.prefetch(1)
        
#         return prefetched


class DataGenerator(tf.keras.utils.Sequence): # for HitNet and (total) ChargeNet
    def __init__(self, 
                 func, # e.g. load_hits
                 dirs=['/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols'], 
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 batch_size=4096,
                 pulses='SRTTWOfflinePulsesDC',
                 geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                 shuffle='free'
                ):
        
        if func == load_charges:
            assert shuffle == 'free', "Only free shuffling for total charges."
        elif func == load_hits:
            assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."
            assert type(pulses) == str, "Use just one type of pulses for HitNet."
        else:
            raise NameError("Currently DataGenerator just supports 'load_hits' and 'load_charges'. 'load_doms' has its own DataGenerator class.")
        
        self.batch_size = int(batch_size/2) # half true labels half false labels
        self.labels = labels
        for i, dir in enumerate(dirs):
            data, params, _ = func(dir=dir, labels=labels, pulses=pulses, geo=geo)
            if i == 0:
                self.data = data
                self.params = params
            else:
                self.data = np.append(self.data, data, axis=0)
                self.params = np.append(self.params, params, axis=0)
        
        if func == load_hits:
            #self.data[:, 3] += -(1.-self.data[:, 5]) * 25. #
            
            #split hits (q=2 hit equal to 2 q=1 hits)
            r = np.floor(self.data[:, 4]) + (self.data[:, 4] - np.floor(self.data[:, 4]) > np.random.rand(len(self.data)))
            self.data = np.repeat(self.data, r.astype(np.int32), axis=0)
            self.params = np.repeat(self.params, r.astype(np.int32), axis=0)
        
            #spread absolute time values
            shifts = np.clip(np.random.normal(0, 1500, len(self.data)), -5000, 5000)
            self.data[:, 3] += shifts
            self.params[:, 3] += shifts
            
            #weight/cut by number of pulses
            #uniques, counts = np.unique(self.params[:, :3], return_counts=True, axis=0)
            #self.weights = 10.0 / np.repeat(counts, counts)
        
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
        '''
        Generate one batch of data
        
        Parameters
        -----------
        index : int
            batch index (between 0 and len(DataGenerator))
        
        Returns
        --------
        X : list 
            the NN input, contains two arrays of length batch_size [observations, params]
        y : array 
            the output the NN should give, has length batch_size
        '''
        
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
        
    def shuffle_params_inDOM(self):
        shuffled_params = np.empty_like(self.params)
        u,c = np.unique(self.data[:, 9], return_counts=True)
        for DOM_index in u[c>1]:
            mask = self.data[:, 9] == DOM_index
            shuffled_params[mask] = np.random.permutation(self.params[mask])
        
        self.shuffled_params = shuffled_params

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        # Generate data similar to Data.get_dataset()
        x = np.take(self.data, indexes_temp, axis=0)
        t = np.take(self.params, indexes_temp, axis=0)
        if len(self.shuffled_params) == 0:
            tr = np.random.permutation(t)
        else:
            tr = np.take(self.shuffled_params, indexes_temp, axis=0)

        d_true_labels = np.ones((self.batch_size, 1), dtype=x.dtype) #* 0.9
        d_false_labels = np.zeros((self.batch_size, 1), dtype=x.dtype)
        
        d_X = np.append(x, x, axis=0)
        d_T = np.append(t, tr, axis=0)
        d_labels = np.append(d_true_labels, d_false_labels)
        
        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return [d_X, d_T], d_labels
    
    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        
        return a[p], b[p], c[p]
    
class DataGenerator_DOMNet(tf.keras.utils.Sequence): # for DOMNet (needs less mem)
    def __init__(self, 
                 dirs=['/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols'], 
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 batch_size=32,
                 container_size=1,
                 reduced=True,
                ):
        
        self.batch_size = int(batch_size/2) # half true labels half false labels
        self.container_size = container_size
        self.labels = labels
        for i, dir in enumerate(dirs):
            if i == 0:
                self.hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
                self.hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
                self.mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
                self.mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
                self.mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))
            else:
                hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
                for i, h in enumerate(hits_idx):
                    hits_idx[i] = (h[0]+len(self.hits), h[1]+len(self.hits))
                self.hits_idx = np.append(self.hits_idx, hits_idx)
                self.hits = np.append(self.hits, np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy')))
                
                mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
                for i, m in enumerate(mctree_idx):
                    mctree_idx[i] = (m[0]+len(self.mctree), m[1]+len(self.mctree))
                self.mctree_idx = np.append(self.mctree_idx, mctree_idx)
                self.mctree = np.append(self.mctree, np.load(os.path.join(dir, 'I3MCTree/data.npy')))
                self.mcprimary = np.append(self.mcprimary, np.load(os.path.join(dir, 'MCInIcePrimary/data.npy')))

        if reduced:
            self.allowed_DOMs = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_DOMs.npy'))
        else:
            self.allowed_DOMs = np.arange(5160)
        geo = np.load(pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'))
        self.doms_blank = np.zeros((self.container_size*self.batch_size,) + (5160, 4,), dtype=np.float32)
        self.doms_blank[:, self.allowed_DOMs, 0:3] = geo.reshape((5160, 3))[self.allowed_DOMs]
        #self.random_state = np.random.permutation(np.arange(2*self.container_size*self.batch_size*len(self.allowed_DOMs)))
        
        self.indexes = np.arange(len(self.hits_idx))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.hits_idx) / self.batch_size))

    def __getitem__(self, index):
        # generate data for some batches
        if index%self.container_size == 0:
            indexes = self.indexes[index*self.batch_size:(index+self.container_size)*self.batch_size]
            self.X_container, self.T_container, self.y_container = self.__data_generation(indexes) #, self.w_container
        
        start = (index%self.container_size)*self.batch_size*2*len(self.allowed_DOMs)
        stop = (index%self.container_size+1)*self.batch_size*2*len(self.allowed_DOMs)
        X = [self.X_container[start:stop], self.T_container[start:stop]]
        y = self.y_container[start:stop]
        #w = self.w_container[start:stop]

        return X, y#, w

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes) # mix between batches

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        
        hits_idx_temp = np.take(self.hits_idx, indexes_temp, axis=0)
        x = self.load_doms(hits_idx_temp, self.hits, len(indexes_temp))
        
        mcprimary_temp = np.take(self.mcprimary, indexes_temp, axis=0)
        mctree_idx_temp = np.take(self.mctree_idx, indexes_temp, axis=0)
        t = get_params(self.labels, mcprimary_temp, self.mctree, mctree_idx_temp)
        tr = np.roll(t, 1, axis=0)
        t = np.repeat(t, repeats=len(x)/len(t), axis=0)
        tr = np.repeat(tr, repeats=len(x)/len(tr), axis=0)

        d_true_labels = np.ones((len(x), 1), dtype=x.dtype)
        d_false_labels = np.zeros((len(x), 1), dtype=x.dtype)
        
        d_X = np.append(x, x, axis=0)
        d_T = np.append(t, tr, axis=0)
        d_labels = np.append(d_true_labels, d_false_labels)

        d_X, d_T, d_labels = shuffle(d_X, d_T, d_labels)
        #R = np.roll(self.random_state, np.random.randint(len(d_X)))[:2*len(indexes_temp)*len(self.allowed_DOMs)]
        
        #weights = np.clip((d_T[:, 6] + d_T[:, 7])/5, 1, 20)

        return d_X, d_T, d_labels#, weights
    
    def load_doms(self, hits_idx, hits, Nevents):
        doms = self.doms_blank.copy()[:Nevents]
        for i in range(len(hits_idx)):
            this_idx = hits_idx[i]
            this_hits = hits[this_idx[0] : this_idx[1]]
            idx = (this_hits['key']['string'] - 1) * 60 + this_hits['key']['om'] - 1
            charges = this_hits['pulse']['charge']
            for j in range(len(this_hits)):
                doms[i, idx[j], 3] += charges[j]

        doms = np.take(doms, self.allowed_DOMs, axis=1)
        return doms.reshape(-1, 4)
