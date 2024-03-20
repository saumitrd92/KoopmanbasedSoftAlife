import tensorflow as tf
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Lambda, Embedding, InputLayer, Concatenate, Reshape, LeakyReLU
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import keras.backend as K
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from packaging import version
import tensorboard
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")


def form_complex_conjugate_block(omegas, delta_t, j):
    scale = tf.exp(omegas[j].output[:, 1] * delta_t)
    entry11 = tf.multiply(scale, tf.cos(omegas[j].output[:, 0] * delta_t))
    entry12 = tf.multiply(scale, tf.sin(omegas[j].output[:, 0] * delta_t))
    row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    L_stack = tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other

    return L_stack

def loss_reconpsi(y_true, y_pred):
    global my_args
    state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder = my_args
    return tf.reduce_mean(tf.square(enc_inputs - reconpsi), axis=-1)

def loss_reconphi(y_true, y_pred):
    global my_args
    state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder = my_args
    return tf.reduce_mean(tf.square(enc_inputs - reconphi), axis=-1)

def loss_recon(y_true, y_pred):
    global my_args
    state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder = my_args
    return (tf.reduce_mean(tf.square(enc_inputs - reconphi), axis=-1) + tf.reduce_mean(tf.square(enc_inputs - reconpsi), axis=-1))*0.5

def loss_linear(y_true, y_pred):
    global my_args
    state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder = my_args
    return tf.reduce_mean(tf.square(psi_encoder(tf.reshape(y_true,[-1,state_features]) + enc_inputs) - linearx), axis=-1)

def loss_pred(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def loss_infinity(y_true, y_pred):
    global my_args
    state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder = my_args
    return tf.reduce_max(tf.abs(y_true - y_pred), axis=-1) + tf.reduce_max(tf.abs(enc_inputs - reconpsi), axis=-1) + tf.reduce_max(tf.abs(enc_inputs - reconphi), axis=-1)


def custom_loss(y_true, y_pred):

    w_recon = 1
    w_lin = 1
    w_pred = 0.1
    w_inf = 10 ** (-5)

    L_recon_phi = loss_reconphi(y_true, y_pred)
    
    L_recon_psi = loss_reconpsi(y_true, y_pred)

    L_linear = loss_linear(y_true, y_pred)
            
    L_pred = loss_pred(y_true, y_pred)

    L_infinity = loss_infinity(y_true, y_pred)

    L_total = w_recon*L_recon_phi + w_recon*L_recon_psi + w_lin*L_linear + w_pred*L_pred + w_inf*L_infinity
    
    return L_total

def lamda_nn_complex(input_features,i):
    nn_input = Input(tensor=input_features, name='complex_lam_{}_input'.format(i+1))
    nn1 = Dense(32, name='complex_lam_{}_dense1'.format(i+1), kernel_regularizer=L1L2(l1=1e-16, l2=1e-18))(nn_input)
    nn11 = LeakyReLU(alpha=0.5)(nn1)
    nn2 = Dense(32, name='complex_lam_{}_dense2'.format(i+1), kernel_regularizer=L1L2(l1=1e-16, l2=1e-18))(nn11)
    nn21 = LeakyReLU(alpha=0.5)(nn2)
    nn5 = Dense(2, name='complex_lam_{}_output'.format(i+1))(nn21)
    nn51 = LeakyReLU(alpha=0.5)(nn5)
    nn = Model(nn_input, nn51)
    return nn

def lamda_nn_real(input_features,i):
    nn_input = Input(tensor=input_features, name='real_lam_{}_input'.format(i+1))
    nn1 = Dense(32, name='real_lam_{}_dense1'.format(i+1), kernel_regularizer=L1L2(l1=1e-16, l2=1e-18))(nn_input)
    nn11 = LeakyReLU(alpha=0.5)(nn1)
    nn2 = Dense(32, name='real_lam_{}_dense2'.format(i+1), kernel_regularizer=L1L2(l1=1e-16, l2=1e-18))(nn11)
    nn21 = LeakyReLU(alpha=0.5)(nn2)
    nn5 = Dense(1, name='real_lam_{}_output'.format(i+1))(nn21)
    nn51 = LeakyReLU(alpha=0.5)(nn5)
    nn = Model(nn_input, nn51)
    return nn

def koopman_block(args):
    
    global num_complex_pairs
    global num_real

    lam_inputs = args

    ### Add Lambda ###
    
    lam_list =[]

    for i in range(num_complex_pairs+num_real):
        if i < num_complex_pairs:
            temp_lam = lamda_nn_complex(lam_inputs[:,2*i:2*i+2],i)
        else:
            temp_lam = lamda_nn_real(lam_inputs[:,i+num_complex_pairs][:, np.newaxis],i)
        lam_list.append(temp_lam)

    delta_t = tf.constant(1.0)

    complex_list = []

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = tf.stack([lam_inputs[:, ind:ind + 2], lam_inputs[:, ind:ind + 2]], axis=2)  # [None, 2, 2]
        L_stack = form_complex_conjugate_block(lam_list, delta_t, j)
        elmtwise_prod = tf.multiply(ystack, L_stack)
        complex_list.append(tf.reduce_sum(elmtwise_prod, axis=1))

    if len(complex_list)>1:
        # each element in list output_list is shape [None, 2]
        complex_part = tf.concat(complex_list,axis=1)
    elif len(complex_list)==1:
        complex_part = complex_list[0]

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp = lam_inputs[:, ind]
        real_list.append(tf.multiply(temp[:, np.newaxis], tf.exp(lam_list[num_complex_pairs + j].output * delta_t)))

    if len(real_list)>1:
        real_part = tf.concat(real_list,axis=1)
    elif len(real_list)==1:
        real_part = real_list[0]

    if len(complex_list) and len(real_list):
        koopman = tf.concat([complex_part, real_part],axis=1)
    elif len(complex_list):
        koopman = complex_part
    else:
        koopman = real_part

    return koopman

def pipeline(params):

    ############################
    ### Initialize variables ###
    ############################
    global num_complex_pairs
    global num_real
    state_features = params['state_features']
    k= params['hp_num_neighbors']
    num_complex_pairs = params['hp_num_complex_pairs']
    num_real = params['hp_num_real']
    h = num_real + 2*num_complex_pairs

    #############################
    ### Pipeline construction ###
    #############################

    ### Add Phi Encoder ###
    phi_enc_input_shape = (state_features,)
    enc_inputs = Input(shape=phi_enc_input_shape, name='phi_enc_input')

    ### Add Beta Network ###
    beta_input_shape = (k*state_features,)
    beta_inputs = Input(shape=beta_input_shape, name='beta_input')
    beta_dense1 = Dense(params['hp_beta_units'], activation="relu", name='beta_dense1', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(beta_inputs)
    # beta_dense2 = Dense(32, activation="relu", name='beta_dense2', kernel_regularizer=L1L2(l1=1e-6, l2=1e-6))(beta_dense1)
    # beta_dense3 = Dense(32, activation="relu", name='beta_dense3', kernel_regularizer=L1L2(l1=1e-6, l2=1e-6))(beta_dense2)
    beta_dense4 = LeakyReLU(alpha=0.5)(Dense(state_features, name='beta_dense4')(beta_dense1))
    # beta_flatten = Flatten()(beta_dense4)

    concat_enc_inputs = Lambda(lambda z: tf.concat(z,axis=-1), name='concat')([enc_inputs,beta_dense4])

    phi_enc_dense1 = Dense(params['hp_phi_enc_units'], activation="relu", name='phi_enc_dense1', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(concat_enc_inputs)
    phi_enc_dense21 = Dense(params['hp_phi_enc_units'], activation="relu", name='phi_enc_dense21', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(phi_enc_dense1)
    # phi_enc_dense22 = Dense(32, activation="relu", name='phi_enc_dense22', kernel_regularizer=L1L2(l1=1e-6, l2=1e-6))(phi_enc_dense21)
    phi_enc_dense3 = LeakyReLU(alpha=0.5)(Dense(h, name='phi_enc_dense3')(phi_enc_dense21))
    phi_encoder = Model([enc_inputs,beta_inputs], phi_enc_dense3, name='phi_Encoder')

    ### Add Psi Encoder ###
    psi_enc_dense1 = Dense(params['hp_psi_enc_units'], activation="relu", name='psi_enc_dense1', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(enc_inputs)
    psi_enc_dense21 = Dense(params['hp_psi_enc_units'], activation="relu", name='psi_enc_dense21', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(psi_enc_dense1)
    # psi_enc_dense22 = Dense(32, activation="relu", name='psi_enc_dense22', kernel_regularizer=L1L2(l1=1e-6, l2=1e-6))(psi_enc_dense21)
    psi_enc_dense3 = LeakyReLU(alpha=0.5)(Dense(h, name='psi_enc_dense3')(psi_enc_dense21))
    psi_encoder = Model(enc_inputs, psi_enc_dense3, name='psi_Encoder')

    ### Add Lambda ###
    lam_input_shape = (h,)
    lam_inputs = Input(shape=lam_input_shape, name='lam_input')

    ### Add Koopman ###
    koopman = Lambda(lambda z: koopman_block(z), name='koopman')(lam_inputs)
    Koopman_Lambda = Model(lam_inputs, koopman, name='Koopman_Lambda')

    ### Add Psi Decoder ###
    psi_dec_input_shape = (h,)
    psi_dec_inputs = Input(shape=psi_dec_input_shape, name='psi_dec_input')
    psi_dec_dense1 = Dense(params['hp_psi_dec_units'], activation="relu", name='psi_dec_dense1', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(psi_dec_inputs)
    psi_dec_dense21 = Dense(params['hp_psi_dec_units'], activation="relu", name='psi_dec_dense21', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(psi_dec_dense1)
    # psi_dec_dense22 = Dense(32, activation="relu", name='psi_dec_dense22', kernel_regularizer=L1L2(l1=1e-6, l2=1e-6))(psi_dec_dense21)
    psi_dec_dense3 = LeakyReLU(alpha=0.5)(Dense(state_features, name='psi_dec_dense3')(psi_dec_dense21))
    psi_decoder = Model(psi_dec_inputs, psi_dec_dense3, name='psi_Decoder')

    ### Add Delta Network ###
    delta_input_shape = (h,)
    delta_inputs = Input(shape=delta_input_shape, name='delta_inputs')
    delta_dense1 = Dense(params['hp_delta_units'], activation="relu", name='delta_dense1', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(delta_inputs)
    delta_dense21 = Dense(params['hp_delta_units'], activation="relu", name='delta_dense21', kernel_regularizer=L1L2(l1=params['hp_l1_reg'], l2=params['hp_l2_reg']))(delta_dense1)
    delta_dense3 = Dense(state_features, activation="sigmoid", name='delta_dense3')(delta_dense21)
    delta_net = Model(delta_inputs, delta_dense3, name='delta_net')

    ### Additional tensors ###
    reconphi = psi_decoder(phi_encoder([enc_inputs,beta_inputs]))
    reconpsi = psi_decoder(psi_encoder(enc_inputs))
    linearx = Koopman_Lambda(phi_encoder([enc_inputs,beta_inputs]))
    predx = delta_net(Koopman_Lambda(phi_encoder([enc_inputs,beta_inputs])))

    global my_args

    my_args = state_features, enc_inputs, reconphi, reconpsi, linearx, psi_encoder

    opt = Adam()

    ### Compile Autoencoder ###
    ae = Model([enc_inputs,beta_inputs], [reconpsi,reconphi], name='AE')
    ae.compile(loss=loss_recon, optimizer=opt)

    ### Compile Prediction Pipeline ###
    pipeline_nn = Model([enc_inputs,beta_inputs], predx, name='Pipeline')

    pipeline_nn.compile(loss=custom_loss, optimizer=opt, metrics=[loss_reconphi,loss_reconpsi,loss_pred,loss_linear,loss_infinity])

    return ae, pipeline_nn