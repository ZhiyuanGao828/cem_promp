import intprim
from intprim.probabilistic_movement_primitives import *
import numpy as np
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
import random
import pandas as pd
from frechetdist import frdist
from cem_promp.cem import NES
from cem_promp.utils import *
# import data
dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')


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
# upper_bound = np.zeros(samples.shape)
# lower_bound = np.zeros(samples.shape)
stdqs = np.zeros(samples.shape)
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    std_q = np.diagonal(Sigma_marg_q) ** 0.5
    stdqs[:,i] = std_q
    mean_margs[:,i] = mu_marg_q
    # upper_bound[:,i] = mu_marg_q + std_q
    # lower_bound[:,i] = mu_marg_q - std_q


ref_x=get_normalization(mean_margs[0])
ref_y=get_normalization(mean_margs[1])              
ref_traj_T=np.array([ref_x,ref_y]).T 

# reward
def get_reward(via_point):
    
#    first get the conditioned model
    old_promp=promp
    t_cond=np.array([0,20,99])
    q_cond =np.zeros([t_cond.shape[0],2])
    q_cond[0]= mean_margs[:,t_cond[0]]
    q_cond[1]= via_point
    q_cond[2]= mean_margs[:,t_cond[2]]

    # t_cond=np.array([0,50])
    # q_cond =np.zeros([t_cond.shape[0],2])
    # q_cond[0]= mean_margs[:,t_cond[0]]
    # q_cond[1]= via_point

    # t_cond=np.array([50])
    # q_cond =np.zeros([t_cond.shape[0],2])
    # q_cond[0]= via_point

    mu_w_cond_rec, Sigma_w_cond_rec=old_promp.get_basis_weight_parameters()

    for i in range(t_cond.shape[0]):
        mu_w_cond_rec, Sigma_w_cond_rec = old_promp.get_conditioned_weights(domain[t_cond[i]], q_cond[i], mean_w=mu_w_cond_rec, var_w=Sigma_w_cond_rec)

    cond_traj = np.zeros([2,100])
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        cond_traj[:,i] = mu_marg_q_con
 
    col=  collision_detect(limit,cond_traj.T)
#     the output of col is 0 or 1
    obs_dis= traj_rect_dist(cond_traj.T,  limit)
    fre_dis= get_f_dist(ref_traj_T,cond_traj)
    reward =  fre_dis  +5 *col + 0.5*obs_dis
    # reward =  fre_dis   +5 *col 

    # cur_reward = curve_dist(mean_margs.T,cond_traj.T )
    # reward =  5 *col+cur_reward
    return reward

# main


dim_theta = 2

# learning rate of NES
alpha = 1e-3

# Population Size
n_samples = 100

# Elite ratio percentage
top_p = 20

# Number of Generations
n_iterations = 30

# Range of values
min_val = 0
max_val = 10

# Plot or not (0 = False, 1 = True)
plot = 0

# Number of Runs
runs = 1

# Algorithm to run in { CEM, NES, CMA_ES }
algorithm = "NES"

# Evaluation function ( 0 = Sphere, 1 = Rastrigin)
function = 0

# Plot output frequency
pause = 0.01

mean = []
best = []
worst = []






limit= np.array([[4,7],[5,8]])

mu_0 = np.array([3.7,7.7])
# mu_0 = np.array([5,5])
sigma_0=  np.ones(2)*4
cem = CEM(get_reward, 2,mu_0 ,sigma_0)

# v= cem. evalGaussian()
v=cem.evalGaussian()
print(v, get_reward(v))



