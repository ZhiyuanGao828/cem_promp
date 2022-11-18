import intprim
from intprim.probabilistic_movement_primitives import *
import matplotlib.pyplot as plt
from cem_promp.cem import CEM
from cem_promp.utils import *
import time
from pytorch_fid.fid_score import calculate_frechet_distance
import argparse

import torch
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')
num_joints =2
basis_model = intprim.basis.GaussianModel(8, 0.1, ["x","y"])
promp = ProMP(basis_model)
Q = dataset.transpose(0,2,1)

for i in range(len(Q)):
	promp.add_demonstration(Q[i])
n_samples = 30
domain = np.linspace(0,1,100)

mean_margs =  np.zeros([2,100])
sigma_s = np.zeros([2,2,100])
# stdqs =  np.zeros([2,100])
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    sigma_s[:,:,i] = Sigma_marg_q
    mean_margs[:,i] = mu_marg_q

# prepared for fid score
mu_1= mean_margs.T.reshape(-1)
sigma_1 = np.zeros([200,200])
for i in range(100):
    sigma_1[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_s[:,:,i]


# state: start, end , obstcle   长和宽为一的正方形 只要确定左下角的x和y

def get_reward(state, action):
    via_point= np.zeros(2)
    via_point[0]=action[0]
    via_point[1]=action[1]
    # via_point[2]=action[2]
    # via_point= action
    old_promp=promp
    via_point = via_point.reshape([-1,2])
    t_cond = np.zeros(2+via_point.shape[0])
    t_cond[0]=0
    t_cond[-1] = 1 
    for i in range(via_point.shape[0]):
        t_cond[i+1]=0.5

    q_cond =np.zeros([2+via_point.shape[0],2])
    q_cond[0]=mean_margs[:,0]
    q_cond[-1]=mean_margs[:,-1]

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

    limit = np.array([[state[0],state[0]+1],[state[1],state[1]+1]])  
    alif =0.75
    # col_rew=0
    col_new=  collision_detect(limit,cond_traj.T)
    col_ori=collision_detect(limit,mean_margs.T)
    # if col_ori==0:
    #     reward = 0
    # elif col_new ==0:
    # col_rew = 10
    mu_2= cond_traj.T.reshape(-1)
    sigma_2 = np.zeros([200,200])
    for i in range(100):
        sigma_2[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_con_array[:,:,i]
    fid_cost= calculate_frechet_distance(mu_1,sigma_1,mu_2,sigma_2)*0.005

    reward = -20

    if col_ori==0: #以前就不撞
        reward =    -fid_cost
    elif col_new ==0:   #以前撞 现在不撞
        reward =   20
        # else: 
        #     reward = 0

        # reward =    -fid_cost +col_rew
    return -reward


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.stds = torch.Tensor([ 0.1,  0.1])
        # nn.Parameter(torch.ones(n_actions))*0.1
        self.actor = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Sigmoid()
            )

    def forward(self, state):
        mu= self.actor(state.float())
        k=torch.Tensor([10,10])    
        dist = MultivariateNormal(loc=mu*k,scale_tril=torch.diag(self.stds) )
        return dist

policy= Actor()


start=mean_margs[:,0]
end=mean_margs[:,-1]
def env_reset():
    states = np.zeros(2)
    states[0]=np.random.uniform(2,6.5)
    states[1]=np.random.uniform(2,7.5)
    limit = np.array([[states[0],states[0]+1],[states[1],states[1]+1]])  
    d1= rect_dist(start,limit)
    d2= rect_dist(end,limit)
    while d1*d2 ==0:
        states[0]=np.random.uniform(2,6.5)
        states[1]=np.random.uniform(2,7.5)
        limit = np.array([[states[0],states[0]+1],[states[1],states[1]+1]])  
        d1= rect_dist(start,limit)
        d2= rect_dist(end,limit)
    return states



def sample_obst(num_ob):
    states=np.zeros([num_ob,2])
    for i in range(num_ob):
        states[i]=env_reset()       
    states = torch.from_numpy(states)
    return states


def nn_score(para):

    limits= sample_obst(1).numpy()

    via_point= []
    score= np.zeros(1)
    update_para_tensor(policy,para)
    for limit in limits:
        via_point.append(policy.actor(torch.from_numpy(limit.astype(np.float32))))
    for i in range(1):
        score[i] = get_reward(limits[i],via_point[i])
        
    return score.mean()
        
        
mu_init= get_para_array(policy)

sigma_0=1

runs= 0

f = open ("/home/zhiyuan/notebook_script/cem_promp/cem_nn.txt",'w') 

t1= time.time()
while runs <2:

    cem = CEM(nn_score ,mu_init ,sigma_0,maxits=50,N=50, Ne=10,fix_time = True) 
    # v= cem. evalGaussian()
    v=cem.evalGaussian_fixedT()
    min_cost= np.min(cem.reward_buf )
    index =np.where(cem.reward_buf ==np.min(cem.reward_buf ))[0][0]
    best_via= cem.state_buf[index,:]

    np.save('stateBuff{run}'.format(run=runs), cem.state_buf)
    print(v, nn_score(v),file = f)
    print('best_iteration ={ite}, min_cost={rew}'.format(ite=index,rew=min_cost),file = f)
    print('best_via= {via}'.format(via=best_via),file = f) 
    print('runs= {run}'.format(run=runs),file = f)
    runs= runs+1

t2= time.time()