import os
import tensorflow as tf
import numpy as np
import pickle
from time import time


class Save(tf.keras.callbacks.Callback):
    """Callback to save models and training history"""
    def __init__(self,
                 save_every=None,
                 path_template=None,
                 batch_stats=False,
                 ):
        '''
        save_every : int
            save the model every n epochs
        path_template : str with format placeholder
            E.g. "models/something_epcoh_%i"
        '''
        if save_every is not None:
            assert path_template is not None
            dirname = os.path.dirname(path_template)
            os.makedirs(dirname, exist_ok=True)
        self.save_every = save_every
        self.path_template = path_template
        self.batch_stats = batch_stats
        self.hist = {}
        self.hist['train_losses'] = []
        self.hist['test_losses'] = []
        self.hist['time'] = []
        if self.batch_stats:
            self.hist['mean_train_losses'] = []
            self.hist['std_train_losses'] = []
            self.hist['mean_test_losses'] = []
            self.hist['std_test_losses'] = []

            def on_train_batch_end(self, batch, logs={}):
                self.batch_train_losses.append(logs.get('loss'))

            def on_test_batch_end(self, batch, logs={}):
                self.batch_test_losses.append(logs.get('loss'))

        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_t = time()
        if self.batch_stats:
            self.batch_train_losses = []
            self.batch_test_losses = []

    def on_epoch_end(self, epoch, logs={}):
        epoch_end_t = time()
        self.hist['train_losses'].append(logs.get('loss'))
        self.hist['test_losses'].append(logs.get('val_loss'))
        self.hist['time'].append((epoch_end_t - self.epoch_start_t) * 1000)

        if self.batch_stats:
            self.hist['mean_train_losses'].append(np.mean(self.batch_train_losses))
            self.hist['std_train_losses'].append(np.std(self.batch_train_losses))
            self.hist['mean_test_losses'].append(np.mean(self.batch_test_losses))
            self.hist['std_test_losses'].append(np.std(self.batch_test_losses))
        
        # Use epoch over all trainings
        epoch = len(self.hist['train_losses'])
        if self.save_every is not None:
            if epoch % self.save_every == 0:
                self.model.save(self.path_template%(epoch) + '_model.hdf5')
                with open(self.path_template%(epoch) + '_history.pkl', 'wb') as fp:
                    pickle.dump(self.hist, fp, protocol=pickle.HIGHEST_PROTOCOL)
                       
