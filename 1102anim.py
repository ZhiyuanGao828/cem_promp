# %matplotlib
import intprim
from intprim.probabilistic_movement_primitives import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import pandas as pd
from frechetdist import frdist
import math
from cem import CEM
from utils import *

dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')

# /home/zhiyuan/notebook_script/state_buff/safe_out_0.3/half

num_joints =2
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
ref3_t=ref_traj_T[1:99:2,:]

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
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        cond_traj[:,i] = mu_marg_q_con
    col=  collision_detect(limit,cond_traj.T)
    obs_dis= traj_rect_dist(cond_traj.T,  limit)
    if obs_dis > 0.3:
        dis_reward = 3
    else:
        dis_reward = -0.5*obs_dis
    fre_dis= get_f_dist(ref_traj_T,cond_traj)
    reward =  fre_dis  +10 *col + dis_reward
    return reward,col,fre_dis,dis_reward

limit= np.array([[4.98,6.02],[2.98,4.02]])

state_buff= np.load('/home/zhiyuan/notebook_script/state_buff/safe_out_0.3/half/stateBuff2.npy')
# print(state_buff.shape)
# state_buff= np.load('/home/zhiyuan/notebook_script/state_buff/safe_in_0.3/half/stateBuff3.npy')

reward = np.zeros([50,6])
col =  np.zeros([50,6])
fre_dis =  np.zeros([50,6])
dis_reward =  np.zeros([50,6])
for i in range(50):
    reward[i,:],col[i,:],fre_dis[i,:],dis_reward[i,:]= get_reward(state_buff[i,:])

def get_traj(via_point):
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
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        cond_traj[:,i] = mu_marg_q_con
    
    return cond_traj

trajs=  np.zeros([50,2,100])
for i in range(50):
    trajs[i,:,:]= get_traj(state_buff[i,:])
    
# np.save('trajs1031', trajs)

via_p=np.zeros([50,4])
via_p[:,0] = state_buff[:,0]
via_p[:,1] = state_buff[:,1]
via_p[:,2] = state_buff[:,3]
via_p[:,3] = state_buff[:,4]

via_x=np.zeros([50,2])
via_y=np.zeros([50,2])
via_x[:,0] = state_buff[:,0]
via_x[:,1] = state_buff[:,3]
via_y[:,0] = state_buff[:,1]
via_y[:,1] = state_buff[:,4]


# from matplotlib.animation import FuncAnimation

# def build_frame2(k):
    
#     x=trajs[k,0,:]
#     y =trajs[k,1,:]
#     p = plt.plot(x, y, color='blue')
#     p += plt.plot(via_x[k,0], via_y[k,0],  marker="o", markersize=5) 
#     p += plt.plot(via_x[k,1], via_y[k,1],  marker="o", markersize=5) 
#     return p

# ite_list = np.arange(0,50)

# fig, ax = plt.subplots()

# ax.axis([0, 12, 0, 12])
# plt.gca().set_aspect(1)
# # for i in range(30):
# #     ax.plot(dataset[i,:,0], dataset[i,:,1],'g-',alpha=0.3)
# ax.add_patch(Rectangle((5, 3), 1, 1,
#              angle=0,
#              edgecolor = 'none',
#              facecolor = 'blue',
#              fill=True,
#              lw=1))
# ax.add_patch(Rectangle((4.8, 2.8), 1.4, 1.4,
#              angle=0,
#              edgecolor = 'black',
#              facecolor = 'blue',
#              fill=False,
#              lw=1))

# ax.plot(mean_margs[0,:], mean_margs[1,:],'g-',alpha=0.3)
# ani = FuncAnimation(fig, build_frame2,frames=ite_list, blit=True)
# plt.show()

# ani.save('1031_01.gif')
# ani.save('1031_01.mp4')