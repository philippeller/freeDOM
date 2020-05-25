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

    def get_hitnet_data(self, train_batch_size=1024, test_batch_size=256, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_hits(dir=dir, labels=self.labels, geo=self.geo))
        single_hits = np.concatenate([d[0] for d in data])
        repeated_params = np.concatenate([d[1] for d in data])
        
        hits_train, hits_test, params_train, params_test = train_test_split(single_hits,
                                                                            repeated_params,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        
        train = self.get_dataset(hits_train, params_train, batch_size=train_batch_size)
        test = self.get_dataset(hits_test, params_test, batch_size=test_batch_size, test=True)
        
        return train, test
        
        
    def get_chargenet_data(self, train_batch_size=1024, test_batch_size=256, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_charges(dir=dir, labels=self.labels))
        total_charge = np.concatenate([d[0] for d in data])
        params = np.concatenate([d[1] for d in data])

        charge_train, charge_test, params_train, params_test = train_test_split(total_charge,
                                                                                params, 
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        
        train = self.get_dataset(charge_train, params_train, batch_size=train_batch_size)
        test = self.get_dataset(charge_test, params_test, batch_size=test_batch_size, test=True)
        
        return train, test
        
    def get_stringnet_data(self, train_batch_size=1024, test_batch_size=256, test_size=0.01, random_state=42):

        data = []
        for dir in self.dirs:
            data.append(load_strings(dir=dir, labels=self.labels))
        string_charges = np.concatenate([d[0] for d in data])
        params = np.concatenate([d[1] for d in data])

        string_train, string_test, params_train, params_test = train_test_split(string_charges,
                                                                                params, 
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        
        train = self.get_dataset(string_train, params_train, batch_size=train_batch_size)
        test = self.get_dataset(string_test, params_test, batch_size=test_batch_size, test=True)
        
        return train, test
        
        
    def get_dataset(self, x, p, batch_size=1024, test=False):
        '''
        get a tensorflow dataset for likelihood approximation

        
        Parameters:
        -----------
        x : ndarray
            observations
        p : ndarray
            parameters        
        batch_size : int
        test : bool
            no shuffling, prefetching and caching
        
        Returns:
        --------
        
        tf.data.Dataset
            with structure ((x, p), y) for training
        
        '''
        
        N = x.shape[0]
        assert p.shape[0] == N
        
        d_x = tf.data.Dataset.from_tensor_slices(x)
        d_p = tf.data.Dataset.from_tensor_slices(p)

        d_true_labels = tf.data.Dataset.from_tensor_slices(np.ones((N, 1), dtype=x.dtype))
        d_false_labels = tf.data.Dataset.from_tensor_slices(np.zeros((N, 1), dtype=x.dtype))

        d_X = tf.data.Dataset.concatenate(d_x, d_x)
        d_P = tf.data.Dataset.concatenate(d_p, d_p.shuffle(N))
        
        d_inputs = tf.data.Dataset.zip((d_X, d_P))        
        d_outputs = tf.data.Dataset.concatenate(d_true_labels, d_false_labels)


        dataset = tf.data.Dataset.zip((d_inputs, d_outputs))
        
        if test:
            return dataset.batch(batch_size)
        
        
        batched = dataset.shuffle(2*N).batch(batch_size=batch_size)
        
        prefetched = batched.prefetch(1).cache()
        
        return prefetched
