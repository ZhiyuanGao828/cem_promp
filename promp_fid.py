import intprim
# import os
# import sys

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
from intprim.probabilistic_movement_primitives import *
import numpy as np
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
import random
import pandas as pd
from frechetdist import frdist
from cem_promp.cem import CEM
from cem_promp.utils import *
import time
from dtw import *
from pytorch_fid.fid_score import calculate_frechet_distance
# import data
dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')
num_joints =2
basis_model = intprim.basis.GaussianModel(8, 0.1, ["x","y"])
promp = ProMP(basis_model)
Q = dataset.transpose(0,2,1)
# Add Demonstrations to the ProMP, which in turn calculates the list of weights for each demonstration.
for i in range(len(Q)):
	promp.add_demonstration(Q[i])
n_samples = 30# Number of trajectoies to sample
domain = np.linspace(0,1,100)

mean_margs =  np.zeros([2,100])
sigma_s = np.zeros([2,2,100])
# stdqs =  np.zeros([2,100])
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    sigma_s[:,:,i] = Sigma_marg_q
    mean_margs[:,i] = mu_marg_q

def get_reward(via_point):
    old_promp=promp
    via_point = via_point.reshape([-1,3])
    t_cond = np.zeros(2+via_point.shape[0])
    t_cond[0]=0
    t_cond[-1] = 1
    for i in range(via_point.shape[0]):
        t_cond[i+1]=via_point[i,2]

    q_cond =np.zeros([2+via_point.shape[0],2])
    q_cond[0]= mean_margs[:,0]
    q_cond[-1]= mean_margs[:,-1]
    for i in range(via_point.shape[0]):
        q_cond[i+1]=via_point[i,:2]

    mu_w_cond_rec, Sigma_w_cond_rec=old_promp.get_basis_weight_parameters()

    for i in range(t_cond.shape[0]):
        mu_w_cond_rec, Sigma_w_cond_rec = old_promp.get_conditioned_weights(t_cond[i], q_cond[i], mean_w=mu_w_cond_rec, var_w=Sigma_w_cond_rec)

    cond_traj = np.zeros([2,100])
    sigma_con_array =np.zeros([2,2,100]) 
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        sigma_con_array[:,:,i] = Sigma_marg_q_con
        cond_traj[:,i] = mu_marg_q_con
    col=  collision_detect(limit,cond_traj.T)
    obs_dis= traj_rect_dist(cond_traj.T,  limit)

    mu_1= mean_margs.T.reshape(-1)
    mu_2= cond_traj.T.reshape(-1)
    sigma_1 = np.zeros([200,200])
    sigma_2 = np.zeros([200,200])
    for i in range(100):
        sigma_1[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_s[:,:,i]
        sigma_2[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_con_array[:,:,i]
    fid_cost= calculate_frechet_distance(mu_1,sigma_1,mu_2,sigma_2) *0.01

    alif =0.001
    # reward =  col   -obs_dis
    # reward =  col + fid_cost 
    reward =  col +alif* fid_cost  -obs_dis*(1-alif)

    return reward

# main
mean10=np.array([[0.,  0., 0.],
 [3.67397766 ,3.8271746 ,0.1],
 [4.45325292 ,4.32710191,0.2],
 [4.96973448, 5.85732177,0.3],
 [4.69102155 ,7.57122257,0.4],
 [3.73612834 ,7.7028387 ,0.5],
 [3.16230054, 6.18850334,0.6],
 [3.69744239 ,4.47427853,0.7],
 [4.8205762  ,3.48503346,0.8],
 [5.67506683 ,3.03704627,0.9]])

limit= np.array([[4.98,6.02],[2.98,4.02]])
sigma_0=  2
runs= 0
f = open ("/home/zhiyuan/notebook_script/cem_promp/multi_via.txt",'w')
mu_0 = np.concatenate([mean10[3,:]  ,mean10[7,:]   ] , axis=0)
t1= time.time()
while runs <1:
    cem = CEM(get_reward ,mu_0 ,sigma_0,maxits=50,N=60, Ne=15,v_min=[0,0,.01], v_max=[15,15,0.99]) 
    v= cem. evalGaussian()
    min_cost= np.min(cem.reward_buf )
    index =np.where(cem.reward_buf ==np.min(cem.reward_buf ))[0][0]
    best_via= cem.state_buf[index,:]

    np.save('stateBuff{run}'.format(run=runs), cem.state_buf)
    print(v, get_reward(v),file = f)
    print('best_iteration ={ite}, min_cost={rew}'.format(ite=index,rew=min_cost),file = f)
    print('best_via= {via}'.format(via=best_via),file = f) 
    print('runs= {run}'.format(run=runs),file = f)
    runs= runs+1

t2= time.time()

