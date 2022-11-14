import intprim
from intprim.probabilistic_movement_primitives import *
import numpy as np
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
import random
import pandas as pd
from frechetdist import frdist
from cem_promp.cem import CEM
from cem_promp.utils import *
# import data
dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')
dataset[:,:,0]=dataset[:,:,0]*2 

# promp
num_joints =2
# Create a ProMP with Gaussian basis functions.

basis_model = intprim.basis.GaussianModel(8, 0.1, ["x","y"])
promp = ProMP(basis_model)
Q = dataset.transpose(0,2,1)

# Add Demonstrations to the ProMP, which in turn calculates the list of weights for each demonstration.
for i in range(len(Q)):
	promp.add_demonstration(Q[i])

n_samples = 30# Number of trajectoies to sample
domain = np.linspace(0,1,100)

# sample some trajs
samples_learn= []
for i in range(n_samples):
    samples, _ = promp.generate_probable_trajectory(domain)
    samples_learn.append(samples)
samples_learn=np.array(samples_learn)  


mean_margs = np.zeros(samples.shape)

stdqs = np.zeros(samples.shape)
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    std_q = np.diagonal(Sigma_marg_q) ** 0.5
    stdqs[:,i] = std_q
    mean_margs[:,i] = mu_marg_q



ref_x=get_normalization(mean_margs[0])
ref_y=get_normalization(mean_margs[1])              
ref_traj_T=np.array([ref_x,ref_y]).T 
limit_0 = np.array([[6,6.5],[6,11]])
limit_1 = np.array([[6,6.5],[0,4]])

# reward
def get_reward(via_point):
    
#    first get the conditioned model
    old_promp=promp

    # t_cond=np.array([30])
    # q_cond =np.zeros([t_cond.shape[0],2])
    # q_cond[0]= via_point

    t_cond=np.array([0,30,99])
    q_cond =np.zeros([t_cond.shape[0],2])
    q_cond[0]= mean_margs[:,t_cond[0]]
    q_cond[1]= via_point
    q_cond[2]= mean_margs[:,t_cond[2]]



    mu_w_cond_rec, Sigma_w_cond_rec=old_promp.get_basis_weight_parameters()

    for i in range(t_cond.shape[0]):
        mu_w_cond_rec, Sigma_w_cond_rec = old_promp.get_conditioned_weights(domain[t_cond[i]], q_cond[i], mean_w=mu_w_cond_rec, var_w=Sigma_w_cond_rec)

    cond_traj = np.zeros([2,100])
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        cond_traj[:,i] = mu_marg_q_con
 
    end_pos = cond_traj[:,-1]
    endcol_1=1
    endcol_0=1
    if 4<end_pos [1]<6:
        endcol_1= 0
    elif  end_pos [0]>7:
        endcol_0 =0
    else:
        endcol_1=1
        endcol_0=1

    end_rew=endcol_0+endcol_1


    col_0=  collision_detect(limit_0,cond_traj.T)
    col_1=  collision_detect(limit_1,cond_traj.T)

    fre_dis= get_f_dist(ref_traj_T,cond_traj)
    reward =  fre_dis  +10 *col_0 +5 *col_1+end_rew
    # reward =  fre_dis 
    return reward

# main
limit= np.array([[5,8],[5,8]])

# mu_0 = np.array([3.7,7.7])
mu_0 = np.array([5,5])
sigma_0=  np.ones(2)*4
cem = CEM(get_reward, 2,mu_0 ,sigma_0)

# v= cem. evalGaussian()
v=cem.evalGaussian()
print(v, get_reward(v))

reward_list=np.zeros(cem.maxits)
index_list= np.linspace(1,cem.maxits,cem.maxits)
for i in range(cem.maxits):
    reward_list[i]= get_reward(cem.state_buf[i])
plt.plot(index_list, reward_list, 'g', alpha=0.8)
label = ['reward']
plt.legend(label, loc='upper left')
plt.savefig('./reward.jpg')
plt.show()

