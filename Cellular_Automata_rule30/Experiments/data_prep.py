import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

def prep_training_data():

    ########################
    ### Training Dataset ###
    ########################

    ### Import Training Dataset ###
    p1, q1, r1, s1 = np.load('sim_data/training/training1.npy')
    p2, q2, r2, s2 = np.load('sim_data/training/training2.npy')
    p3, q3, r3, s3 = np.load('sim_data/training/training3.npy')
    p4, q4, r4, s4 = np.load('sim_data/training/training4.npy')
    p5, q5, r5, s5 = np.load('sim_data/training/training5.npy')
    p6, q6, r6, s6 = np.load('sim_data/training/training6.npy')
    p7, q7, r7, s7 = np.load('sim_data/training/training7.npy')
    p8, q8, r8, s8 = np.load('sim_data/training/training8.npy')
    p9, q9, r9, s9 = np.load('sim_data/training/training9.npy')
    p10, q10, r10, s10 = np.load('sim_data/training/training10.npy')

    p = np.concatenate((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10))
    q = np.concatenate((q1,q2,q3,q4,q5,q6,q7,q8,q9,q10))
    r = np.concatenate((r1,r2,r3,r4,r5,r6,r7,r8,r9,r10))
    s = np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10))

    x = q
    x_neigh = np.transpose(np.stack([p,r]))
    x_dot = (s!=q).astype(int) # Flip/Spin (1) or no Flip/Spin (0)
    x_plus = s

    np.save('sim_data/training/x.npy',x)
    np.save('sim_data/training/x_neigh.npy',x_neigh)
    np.save('sim_data/training/x_plus.npy',x_plus)
    np.save('sim_data/training/x_dot.npy',x_dot)

