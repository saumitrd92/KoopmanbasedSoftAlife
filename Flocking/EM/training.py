import os
from datetime import datetime
import tensorboard
from keras.callbacks import TensorBoard
import numpy as np
import network as net
import data_prep as dp
import keras_tuner
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str)
args = parser.parse_args()
exp_name = args.experiment

def training_code(params):
    ################################
    ### Preprocess Training Data ###
    ################################

    if not os.path.exists(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_head.npy'):
        if not os.path.exists(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/'):
            os.mkdir(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope']))
        dp.prep_training_data(params)

    # Define the Keras TensorBoard callback.
    dt_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir=params['exp_path']+"logs/fit/" + dt_stamp
    params['logdir'] = logdir
    tensorboard_callback = TensorBoard(log_dir=params['logdir'])

    # Load training data 
    train_x = np.load(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_head.npy')
    train_x_neigh = np.load(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_neigh_head.npy')
    train_x_plus = np.load(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_plus_head.npy')
    train_x_dot = np.load(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_dot_head.npy')
    train_x_dot_max = np.load(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_dot_head_max.npy')

    ae, pipeline = net.pipeline(params)

    ############################
    ### Training Autoencoder ###
    ############################

    ae.fit([train_x_neigh,train_x], train_x, epochs=500, batch_size=1000, verbose=0,callbacks=[tensorboard_callback])


    # ae.save_weights(params['exp_path']+"Realizations/" + dt_stamp +'_ae.h5')


    #########################
    ### Training Pipeline ###
    #########################

    history = pipeline.fit([train_x_neigh,train_x], train_x_dot, epochs=2000, batch_size=1000, verbose=0,shuffle=True,callbacks=[tensorboard_callback], validation_split=0.2)

    pipeline.save_weights(params['exp_path']+"Realizations/" + dt_stamp +'_pipeline.h5')

    validation_loss = history.history['val_loss'][-1]
    return validation_loss

class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters

        params={}

        ### Experiment Parameters ###
        params['exp_path'] = exp_name+'/'
        params['state_features'] = 1
        params['group_features'] = 3

        ### Hyper Parameters ###
        params['hp_num_neighbors'] = 50
        params['hp_neigh_envelope'] = hp.Choice("hp_neigh_envelope", [3,5,10])

        params['hp_num_complex_pairs'] = hp.Int("hp_num_complex_pairs", min_value=1, max_value=5, step=1)
        params['hp_num_real'] = hp.Int("hp_num_real", min_value=1, max_value=5, step=1)

        params['hp_beta_units'] = hp.Choice("hp_beta_units", [8,16])

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
    max_trials=30, overwrite=True, directory=exp_name+'/tmp', project_name=exp_name,
)

tuner.search()

# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp)
# training_code(**best_hp.values, saving_path=exp_name+"/tmp/best_model")