import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

def prep_training_data():

    ########################
    ### Training Dataset ###
    ########################

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    ### Import Training Dataset ###
    cell1, neigh1, fs1 = np.load('sim_data/training/training1.npy')
    cell2, neigh2, fs2 = np.load('sim_data/training/training2.npy')
    cell3, neigh3, fs3 = np.load('sim_data/training/training3.npy')
    cell4, neigh4, fs4 = np.load('sim_data/training/training4.npy')
    cell5, neigh5, fs5 = np.load('sim_data/training/training5.npy')
    cell6, neigh6, fs6 = np.load('sim_data/training/training6.npy')
    cell7, neigh7, fs7 = np.load('sim_data/training/training7.npy')
    cell8, neigh8, fs8 = np.load('sim_data/training/training8.npy')
    cell9, neigh9, fs9 = np.load('sim_data/training/training9.npy')
    cell10, neigh10, fs10 = np.load('sim_data/training/training10.npy')

    # restore np.load for future normal usage
    np.load = np_load_old

    cell = np.concatenate((cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8,cell9,cell10))
    neigh = np.concatenate((neigh1[0],neigh2[0],neigh3[0],neigh4[0],neigh5[0],neigh6[0],neigh7[0],neigh8[0],neigh9[0],neigh10[0]))
    fs = np.concatenate((fs1,fs2,fs3,fs4,fs5,fs6,fs7,fs8,fs9,fs10))

    x = cell
    x_neigh = neigh #np.transpose(neigh)
    x_dot = (fs!=cell).astype(int) # Flip/Spin (1) or no Flip/Spin (0)
    x_plus = fs

    np.save('sim_data/training/x.npy',x)
    np.save('sim_data/training/x_neigh.npy',x_neigh)
    np.save('sim_data/training/x_plus.npy',x_plus)
    np.save('sim_data/training/x_dot.npy',x_dot)

