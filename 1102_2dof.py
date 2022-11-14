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

from two_dof.toy_robot import ToyRobot
from two_dof.toy_obstacle import ToyObstacle

dataset= np.load('/home/zhiyuan/notebook_script/cem_promp/data2dof.npy')
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


upper_bound =  np.zeros([2,100])
lower_bound =  np.zeros([2,100])
sigma_s = np.zeros([2,2,100])
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    sigma_s[:,:,i] = Sigma_marg_q
    std_q = np.diagonal(Sigma_marg_q) ** 0.5

    mean_margs[:,i] = mu_marg_q
    upper_bound[:,i] = mu_marg_q + std_q
    lower_bound[:,i] = mu_marg_q - std_q


ref_traj_T= np.load('/home/zhiyuan/notebook_script/ref_trajxy_T.npy')
ref3_t=ref_traj_T[1:99:3,:]
# print(ref_traj_T.shape) #100,2
toy_robo = ToyRobot(6., 8., 0., 0.)

# mean10=np.array([[0.,  0., 0.],
#  [3.67397766 ,3.8271746 ,0.1],
#  [4.45325292 ,4.32710191,0.2],
#  [4.96973448, 5.85732177,0.3],
#  [4.69102155 ,7.57122257,0.4],
#  [3.73612834 ,7.7028387 ,0.5],
#  [3.16230054, 6.18850334,0.6],
#  [3.69744239 ,4.47427853,0.7],
#  [4.8205762  ,3.48503346,0.8],
#  [5.67506683 ,3.03704627,0.9]])

# limit= np.array([[4,7],[5,8]])
# limit= np.array([[5,6],[3,4]])
limit= np.array([[4.98,6.02],[2.98,4.02]])
# mu_0 = np.concatenate([mean10[3,:],mean10[7,:]   ] , axis=0)
# mu_0 = mean10[7,:]
sigma_0=  0.5


# toy_obstacle = ToyObstacle(5.5, 3.5, 1, 1,safe_distance=0.2)
def get_reward(via_point):
#     task space转为joint space
#     v_joint = toy_robo.ik(dataset[i,:,0], dataset[i,:,1])[0]
    
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

    cond_mean = np.zeros([2,100])
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        cond_mean[:,i] = mu_marg_q_con
    
    
    x, y = toy_robo.fk(cond_mean[0,:], cond_mean[1,:])
    cond_traj = np.array([x,y])
    # distance = toy_obstacle.eval_distance(x,y) 
        
        
    
    col=  collision_detect(limit,cond_traj.T)
    obs_dis= traj_rect_dist(cond_traj.T,  limit)
    if obs_dis > 0.3:
        dis_reward = 3
    else:
        dis_reward = -0.5*obs_dis #0.2 is good for dtw


    # if obs_dis > 0.3:
    #     dis_reward = 3
    # # elif obs_dis < 0.1:
    # #     dis_reward = 3
    # else:
    #     dis_reward =0.5*obs_dis #0.2 is good for dtw

    # cond_traj3 = cond_traj[:,1:99:2]    
    cond_traj3 = cond_traj[:,1:99:3]
    # fre_dis= get_f_dist(ref_traj_T,cond_traj)

    fre_dis= get_f_dist(ref3_t,cond_traj3)

    # dtw_cost= get_dtw_cost(ref3_t,cond_traj3)
    reward =  fre_dis  +10 *col + dis_reward
    # reward =  dtw_cost  +10 *col + dis_reward
    # reward =  dtw_cost  +10 *col
    # reward =  fre_dis   + dis_reward
    # reward =  fre_dis   +10*col 
    return reward
    # return reward, fre_dis,col,dis_reward


f = open ("/home/zhiyuan/notebook_script/cem_promp/two_dof.txt",'w') 
# mu_0 = np.concatenate([mean_margs[:,30]  ,mean_margs[:,70]   ] , axis=0)
mu_0 = np.array([-0.37431626,  2.00549582,0.3,-0.60741837,  2.32964738,0.7])
runs= 0

while runs <1:
    # mu_0 = mean10[3,:]
    # mu_0 = np.concatenate([mean10[3,:] ,mean10[7,:]   ] , axis=0)
    cem = CEM(get_reward ,mu_0 ,sigma_0,maxits=50,N=60, Ne=15,v_min=[-3.15,-3.15,.01], v_max=[3.15,3.15,0.99]) 
    v= cem. evalGaussian()
    # v=cem.evalGaussian_fixedT()
    min_cost= np.min(cem.reward_buf )
    index =np.where(cem.reward_buf ==np.min(cem.reward_buf ))[0][0]
    best_via= cem.state_buf[index,:]

    np.save('stateBuff{run}'.format(run=runs), cem.state_buf)
    print(v, get_reward(v),file = f)
    print('best_iteration ={ite}, min_cost={rew}'.format(ite=index,rew=min_cost),file = f)
    print('best_via= {via}'.format(via=best_via),file = f) 
    print('runs= {run}'.format(run=runs),file = f)
    runs= runs+1

