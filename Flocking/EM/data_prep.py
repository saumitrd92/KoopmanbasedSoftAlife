import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
from joblib import Parallel, delayed

# def data_prep_x(agent,neigbours,nb_dist,db,time_step,k):
#     temp = []
#     db_filtered = db.loc[db['Time'] == time_step]
#     # agent_x = db_filtered.loc[db_filtered['Agent'] == agent]['X'].values
#     # agent_y = db_filtered.loc[db_filtered['Agent'] == agent]['Y'].values
#     agent_head = db_filtered.loc[db_filtered['Agent'] == agent]['Heading'].values[0]
#     temp.append(np.asarray([agent_head,0]))#temp.append(np.asarray([agent_x,agent_y,agent_head]))
#     if (neigbours.size>=1):
#         for i in range(neigbours.size):
#             # neigh_x = db_filtered.loc[db_filtered['Agent'] == i]['X'].values
#             # neigh_y = db_filtered.loc[db_filtered['Agent'] == i]['Y'].values
#             neigh_head = db_filtered.loc[db_filtered['Agent'] == neigbours[i]]['Heading'].values[0]
#             temp.append(np.asarray([neigh_head,nb_dist[i]]))#temp.append(np.asarray([neigh_x,neigh_y,neigh_head]))
#         average_head = np.mean(temp)
#         for j in range(k-neigbours.size):
#             temp.append(np.asarray([[[average_head]]]))
#     else:
#         for i in range(k):
#             temp.append(np.asarray([agent_head]))

#     return np.asarray(temp)

def data_prep_x_group(agent,neigbours,nb_dist,db,time_step,k=50,p=5):
    temp = []
    db_filtered = db.loc[db['Time'] == time_step]
    agent_head = db_filtered.loc[db_filtered['Agent'] == agent]['Heading'].values[0]
    if (neigbours.size>=1):
        for i in range(neigbours.size):
            neigh_distance = nb_dist[i]
            neigh_head = db_filtered.loc[db_filtered['Agent'] == neigbours[i]]['Heading'].values[0]
            if neigh_distance <= 0.0142857142857143*p: # Len of p patches
                temp.append(np.asarray([np.sin(2*np.pi*neigh_head)+1, neigh_distance]))
        if len(temp) > 0:
            mean_head = np.mean(temp,axis=0)
            pol = np.sqrt((np.sum(np.array(temp)[:,0]-1))**2 + (np.sum(np.cos(np.arcsin(np.array(temp)[:,0]-1))))**2)/len(temp)
        else:
            mean_head = np.array([np.sin(2*np.pi*agent_head)+1,1.0])
            temp.append(mean_head)
            pol = np.array(1.0)
    else:
        print('No neigbours')

    return np.concatenate((agent_head,mean_head,pol),axis=None)

# def data_prep_x(agent,neigbours,nb_dist,db,time_step,k=10,p=5):
#     temp = []
#     db_filtered = db.loc[db['Time'] == time_step]
#     # agent_x = db_filtered.loc[db_filtered['Agent'] == agent]['X'].values
#     # agent_y = db_filtered.loc[db_filtered['Agent'] == agent]['Y'].values
#     agent_head = db_filtered.loc[db_filtered['Agent'] == agent]['Heading'].values[0]
#     temp.append(np.asarray([agent_head,0,1e4]))#temp.append(np.asarray([agent_x,agent_y,agent_head]))
#     if (neigbours.size>=1):
#         for i in range(neigbours.size):
#             # neigh_x = db_filtered.loc[db_filtered['Agent'] == i]['X'].values
#             # neigh_y = db_filtered.loc[db_filtered['Agent'] == i]['Y'].values
#             neigh_head = db_filtered.loc[db_filtered['Agent'] == neigbours[i]]['Heading'].values[0]
#             neigh_distance = nb_dist[i]
#             if neigh_distance > 0.0142857142857143*p: # Len of p patches
#                 neigh_distance = 999
#             temp.append(np.asarray([neigh_head,neigh_distance,1/(neigh_distance+1e-4)]))#temp.append(np.asarray([neigh_x,neigh_y,neigh_head]))
#         # average_head = np.mean(temp)
#         # for j in range(k-neigbours.size):
#         #     temp.append(np.asarray([[[average_head]]]))
#     else:
#         for i in range(k):
#             temp.append(np.asarray([agent_head]))

#     return np.asarray(temp)

def data_prep_x_plus(agent,db,time_step):
    db_filtered = db.loc[db['Time'] == time_step]
    # agent_x = db_filtered.loc[db_filtered['Agent'] == agent]['X'].values
    # agent_y = db_filtered.loc[db_filtered['Agent'] == agent]['Y'].values
    agent_head = db_filtered.loc[db_filtered['Agent'] == agent]['Heading'].values

    return np.asarray([agent_head])#np.asarray([agent_x,agent_y,agent_head])

def data_prep_x_dot(agent,db,time_step):
    db_filtered_2 = db.loc[db['Time'] == time_step]
    db_filtered_1 = db.loc[db['Time'] == time_step-1]
    # agent_x = db_filtered.loc[db_filtered['Agent'] == agent]['X'].values
    # agent_y = db_filtered.loc[db_filtered['Agent'] == agent]['Y'].values
    agent_head_dot = db_filtered_2.loc[db_filtered_2['Agent'] == agent]['Heading'].values - db_filtered_1.loc[db_filtered_1['Agent'] == agent]['Heading'].values

    return np.asarray([agent_head_dot])#np.asarray([agent_x,agent_y,agent_head])

def my_func(case):
    x = []
    x_plus = []
    x_dot = []

    global df_scaled
    global k

    # for case in df_scaled['Case'].unique():
    print('###########################     CASE   ',case,'   ###########################')
    nb_timesteps = df_scaled.loc[df_scaled['Case'] == case]['Time'].unique().shape[0]
    nb_agents = df_scaled.loc[df_scaled['Case'] == case]['Agent'].unique().shape[0]
    for ts in range(nb_timesteps-1):
        flock = df_scaled.loc[(df_scaled['Case'] == case) & (df_scaled['Time']==ts)][['X','Y']]
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(flock.values)
        distances, indices = nbrs.kneighbors(flock.values)
        for agent in range(nb_agents):
            # neigbours = identify_neighbours(agent,df_master,ts,max_dist=7)
            neigbours = indices[agent][1:]
            nb_dist = distances[agent][1:]
            x.append(data_prep_x_group(agent,neigbours,nb_dist,df_scaled.loc[df_scaled['Case'] == case],ts,k)) # Shape [?,k+1,nb_features]
            x_plus.append(data_prep_x_plus(agent,df_scaled.loc[df_scaled['Case'] == case],ts+1)) # Shape [?,nb_features]
            x_dot.append(data_prep_x_dot(agent,df_scaled.loc[df_scaled['Case'] == case],ts+1))
        print('Timesteps',ts+1,'out of',nb_timesteps-1,'completed')
    return x,x_plus,x_dot

def prep_training_data(params):

    global df_scaled
    global k

    k=params['hp_num_neighbors']

    ########################
    ### Training Dataset ###
    ########################

    ### Import Training Dataset ###

    df_master1 = pd.read_csv(params['exp_path']+'sim_data/training/training1.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master1['Case'] = [1]*df_master1.shape[0]
    df_master2 = pd.read_csv(params['exp_path']+'sim_data/training/training2.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master2['Case'] = [2]*df_master2.shape[0]
    df_master3 = pd.read_csv(params['exp_path']+'sim_data/training/training3.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master3['Case'] = [3]*df_master3.shape[0]
    df_master4 = pd.read_csv(params['exp_path']+'sim_data/training/training4.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master4['Case'] = [4]*df_master4.shape[0]
    df_master5 = pd.read_csv(params['exp_path']+'sim_data/training/training5.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master5['Case'] = [5]*df_master5.shape[0]
    df_master6 = pd.read_csv(params['exp_path']+'sim_data/training/training6.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master6['Case'] = [6]*df_master6.shape[0]
    df_master7 = pd.read_csv(params['exp_path']+'sim_data/training/training7.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master7['Case'] = [7]*df_master7.shape[0]
    df_master8 = pd.read_csv(params['exp_path']+'sim_data/training/training8.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master8['Case'] = [8]*df_master8.shape[0]
    df_master9 = pd.read_csv(params['exp_path']+'sim_data/training/training9.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master9['Case'] = [9]*df_master9.shape[0]
    df_master10 = pd.read_csv(params['exp_path']+'sim_data/training/training10.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    df_master10['Case'] = [10]*df_master10.shape[0]
    df_master = pd.concat([df_master1,df_master2,df_master3,df_master4,df_master5,df_master6,df_master7,df_master8,df_master9,df_master10],ignore_index=True)

    df_master = df_master.loc[df_master.Time < 100]
    # print(df_master.describe())

    # df_master = pd.read_csv(params['exp_path']+'sim_data/training/training1.csv', sep=' ',names=['Time', 'Agent', 'X', 'Y', 'Heading'])
    # df_master['Case'] = [1]*df_master.shape[0]

    ### Scale Training Dataset ###
    X_min = -35.5
    X_max = 35.5
    Y_min = -35.5
    Y_max = 35.5
    Heading_min = 0
    Heading_max = 360

    df_scaled = df_master.copy()
    df_scaled['X'] = ((df_master['X'] - X_min)/(X_max - X_min)).values
    df_scaled['Y'] = ((df_master['Y'] - Y_min)/(Y_max - Y_min)).values
    heads = []
    for i in df_master['Heading'].values:
        if 90 - i != abs(90 - i):
            heads.append(450 - i)
        else:
            heads.append(90 - i)

    df_scaled.loc[:,'Heading'] = heads
    df_scaled.loc[:,'Heading'] = ((df_scaled.loc[:,'Heading'] - Heading_min)/(Heading_max - Heading_min)).values

    # print(df_scaled.describe())

    # x = []
    # x_plus = []
    # x_dot = []

    # for case in df_scaled['Case'].unique():
    #     print('###########################     CASE   ',case,'   ###########################')
    #     nb_timesteps = df_scaled.loc[df_scaled['Case'] == case]['Time'].unique().shape[0]
    #     nb_agents = df_scaled.loc[df_scaled['Case'] == case]['Agent'].unique().shape[0]
    #     for ts in range(nb_timesteps-1):
    #         flock = df_scaled.loc[(df_scaled['Case'] == case) & (df_scaled['Time']==ts)][['X','Y']]
    #         nbrs = NearestNeighbors(n_neighbors=params['hp_num_neighbors']+1, algorithm='ball_tree').fit(flock.values)
    #         distances, indices = nbrs.kneighbors(flock.values)
    #         for agent in range(nb_agents):
    #             # neigbours = identify_neighbours(agent,df_master,ts,max_dist=7)
    #             neigbours = indices[agent][1:]
    #             nb_dist = distances[agent][1:]
    #             x.append(data_prep_x_group(agent,neigbours,nb_dist,df_scaled.loc[df_scaled['Case'] == case],ts,p=params['hp_neigh_envelope'])) # Shape [?,k+1,nb_features]
    #             x_plus.append(data_prep_x_plus(agent,df_scaled.loc[df_scaled['Case'] == case],ts+1)) # Shape [?,nb_features]
    #             x_dot.append(data_prep_x_dot(agent,df_scaled.loc[df_scaled['Case'] == case],ts+1))
    #         print('Timesteps',ts+1,'out of',nb_timesteps-1,'completed')
        
    x,x_plus,x_dot = zip(*Parallel(n_jobs=8,require='sharedmem')(delayed(my_func)(x) for x in df_scaled['Case'].unique()))

    x = x[0]
    x_plus = x_plus[0]
    x_dot = x_dot[0]

    for i in range(len(x_dot)):
        if abs(x_dot[i]) > 0.5:
            if x_dot[i] > 0:
                x_dot[i] -= 1
            else:
                x_dot[i] += 1

    x_head = np.reshape(np.asarray(x)[:,0],(np.asarray(x)[:,0].shape[0],-1))
    x_neigh_head = np.asarray(x)[:,1:]
    x_plus_head = np.asarray(x_plus)[:,:,0]
    x_dot_head = np.asarray(x_dot)[:,:,0]
    x_dot_head_max = np.max(x_dot_head)
    x_dot_head = x_dot_head/x_dot_head_max



    np.save(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_head.npy',x_head)
    np.save(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_neigh_head.npy',x_neigh_head)
    np.save(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_plus_head.npy',x_plus_head)
    np.save(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_dot_head.npy',x_dot_head)
    np.save(params['exp_path']+'sim_data/training/nb_neigh_'+str(params['hp_neigh_envelope'])+'/x_dot_head_max',x_dot_head_max)