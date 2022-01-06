#!/usr/bin/env python
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from freedom.utils.callback import Save
from freedom.utils.dataset import DataGenerator #, DataGenerator_DOMNet
from freedom.utils.i3cols_dataloader import load_hits, load_charges #, load_doms, load_strings, load_layers
#from freedom.neural_nets.transformations import domnet_trafo
from freedom.neural_nets.hitnet import get_hitnet
from freedom.neural_nets.chargenet import get_chargenet
#from freedom.neural_nets.stringnet import get_all_stringnet
#from freedom.neural_nets.layernet import get_layernet
#from freedom.neural_nets.domnet import get_domnet
from freedom.neural_nets.transformations import hitnet_trafo, chargenet_trafo


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

labels = ['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy']

strategy = tf.distribute.MirroredStrategy()
nGPUs = strategy.num_replicas_in_sync


#'/tf/localscratch/weldert/i3cols/120000_i3cols_train/', 
#'/tf/localscratch/weldert/i3cols/120000_i3cols_train2/', 
train_d = ['/tf/localscratch/weldert/i3cols/140000_i3cols_train/',
           '/tf/localscratch/weldert/i3cols/140000_i3cols_train2_1/', '/tf/localscratch/weldert/i3cols/140000_i3cols_train2_2/']
valid_d = ['/tf/localscratch/weldert/i3cols/120000_i3cols_valid/', '/tf/localscratch/weldert/i3cols/140000_i3cols_valid/']

#geo = '/tf/localscratch/weldert/freeDOM/freedom/resources/geo_array.npy'
#geo_upgrade = '/tf/localscratch/weldert/freeDOM/freedom/resources/geo_array_upgrade.npy'
#pulses = 'IceCubePulses' #['IceCubePulsesTWSRT', 'mDOMPulsesTWSRT', 'DEggPulsesTWSRT']

training_generator = DataGenerator(load_hits, train_d, labels, 4096*nGPUs) #, pulses, geo_upgrade
validation_generator = DataGenerator(load_hits, valid_d, labels, 4096*nGPUs)
#training_generator = DataGenerator_DOMNet(train_d, labels, batch_size=8*nGPUs, container_size=20, reduced=True)
#validation_generator = DataGenerator_DOMNet(valid_d, labels, batch_size=8*nGPUs, container_size=20, reduced=True)


# remove events with more than 999 hits or less than 4
uniques, indices, counts = np.unique(training_generator.params[:, :3], return_inverse=True, return_counts=True, axis=0)
counts = counts[indices]
training_generator.data = training_generator.data[(counts < 1000) & (counts > 3)]
training_generator.params = training_generator.params[(counts < 1000) & (counts > 3)]
training_generator.indexes = np.arange(len(training_generator.data))

uniques, indices, counts = np.unique(validation_generator.params[:, :3], return_inverse=True, return_counts=True, axis=0)
counts = counts[indices]
validation_generator.data = validation_generator.data[(counts < 1000) & (counts > 3)]
validation_generator.params = validation_generator.params[(counts < 1000) & (counts > 3)]
validation_generator.indexes = np.arange(len(validation_generator.data))


#optimizer = tf.keras.optimizers.Adam(1e-4) #lr_schedule
radam = tfa.optimizers.RectifiedAdam(lr=1e-4)
optimizer = tfa.optimizers.Lookahead(radam)
with strategy.scope():
    #net = get_chargenet(labels, n_inp=2, activation='swish', final_activation='exponential', use_nCh=True)
    net = get_hitnet(labels, activation='swish', final_activation='exponential')
    #net = get_domnet(labels, activation=tfa.activations.mish)
    '''
    net = tf.keras.models.load_model(
        '../../freedom/resources/models/DeepCore/HitNet_ranger_15_Oct_2021-19h00/epoch_45_model.hdf5', 
        custom_objects={'hitnet_trafo':hitnet_trafo}
    )
    '''
    net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


train_id = 'HitNet_ranger_' + datetime.datetime.now().strftime("%d_%b_%Y-%Hh%M")

callbacks = []
callbacks.append(Save(save_every=2, path_template='../../freedom/resources/models/'+train_id+'/epoch_%i', ))
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='../../freedom/resources/logs/'+train_id, histogram_freq=1))


hist = net.fit(training_generator, epochs=50, verbose=0, callbacks=callbacks, validation_data=validation_generator,
               initial_epoch=len(callbacks[0].hist['train_losses']))
