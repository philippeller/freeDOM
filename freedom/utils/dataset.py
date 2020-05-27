"""Module to create tf.data.DataSet instances for training"""
import pkg_resources
import numpy as np
import tensorflow as tf
from freedom.utils.i3cols_dataloader import load_hits, load_charges, load_strings
from sklearn.model_selection import train_test_split


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
