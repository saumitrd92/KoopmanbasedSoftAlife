import os
from datetime import datetime
import tensorboard
from keras.callbacks import TensorBoard
import numpy as np
import network as net
import data_prep as dp
import keras_tuner
import argparse

def training_code(params):
    ################################
    ### Preprocess Training Data ###
    ################################

    dp.prep_training_data()

    # Define the Keras TensorBoard callback.
    dt_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir="logs/fit/" + dt_stamp
    params['logdir'] = logdir
    tensorboard_callback = TensorBoard(log_dir=params['logdir'])

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # Load training data 
    train_x = np.load('sim_data/training/x.npy')
    train_x_neigh = np.load('sim_data/training/x_neigh.npy')
    train_x_plus = np.load('sim_data/training/x_plus.npy')
    train_x_dot = np.load('sim_data/training/x_dot.npy')

    # restore np.load for future normal usage
    np.load = np_load_old

    ae, pipeline = net.pipeline(params)

    ############################
    ### Training Autoencoder ###
    ############################

    ae.fit([train_x,train_x_neigh], [train_x,train_x], epochs=200, batch_size=1000, verbose=0,callbacks=[tensorboard_callback])


    # ae.save_weights("Realizations/" + dt_stamp +'_ae.h5')


    #########################
    ### Training Pipeline ###
    #########################

    history = pipeline.fit([train_x,train_x_neigh], train_x_dot, epochs=5000, batch_size=1000, verbose=0,shuffle=True,callbacks=[tensorboard_callback], validation_split=0.2)

    pipeline.save_weights("Realizations/" + dt_stamp +'_pipeline.h5')

    validation_loss = history.history['val_loss'][-1]
    return validation_loss

class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters

        params={}

        ### Experiment Parameters ###

        params['state_features'] = 1

        ### Hyper Parameters ###
        params['hp_num_neighbors'] = 8

        params['hp_num_complex_pairs'] = hp.Int("hp_num_complex_pairs", min_value=1, max_value=4, step=2)
        params['hp_num_real'] = hp.Int("hp_num_real", min_value=0, max_value=4, step=2)

        params['hp_beta_units'] = hp.Choice("hp_beta_units", [4,8])

        enc_dec_size = hp.Choice("hp_enc_dec_size", [8,16])
        params['hp_phi_enc_units'] = enc_dec_size
        params['hp_psi_enc_units'] = enc_dec_size
        params['hp_psi_dec_units'] = enc_dec_size

        params['hp_delta_units'] = hp.Choice("hp_delta_units", [8,16])

        l1_reg = hp.Float("hp_l1_reg", min_value=1e-16, max_value=1e-10, sampling="log")
        params['hp_l1_reg'] = l1_reg
        params['hp_l2_reg'] = l1_reg*1e-2

        return training_code(params)


tuner = MyTuner(
    max_trials=10, overwrite=False, directory='tmp')

tuner.search()

# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)

params_best={}

### Experiment Parameters ###

params_best['state_features'] = 1

### Hyper Parameters ###
params_best['hp_num_neighbors'] = 8

params_best['hp_num_complex_pairs'] = best_hp.values["hp_num_complex_pairs"]
params_best['hp_num_real'] = best_hp.values["hp_num_real"]

params_best['hp_beta_units'] = best_hp.values["hp_beta_units"]

enc_dec_size = best_hp.values["hp_enc_dec_size"]
params_best['hp_phi_enc_units'] = enc_dec_size
params_best['hp_psi_enc_units'] = enc_dec_size
params_best['hp_psi_dec_units'] = enc_dec_size

params_best['hp_delta_units'] = best_hp.values["hp_delta_units"]

l1_reg = best_hp.values["hp_l1_reg"]
params_best['hp_l1_reg'] = l1_reg
params_best['hp_l2_reg'] = l1_reg*1e-2

training_code(params_best)