import numpy as np
import pandas as pd
# from frechetdist import frdist
import math
# from pytorch_fid.fid_score import calculate_frechet_distance
# from two_dof.toy_obstacle import ToyObstacle
# from dtw import *

#  traj dim=[100,2]
import torch
import torch.nn as nn


def collision_detect(rect,traj):
    if any(rect[0,0]<=p[0]<=rect[0,1] and rect[1,0]<=p[1]<=rect[1,1] for p in traj):
        col =1 
    else:
        col=0
        
    return col


def get_normalization(data): #data is an 1-dimensional array
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_f_dist(ref_T,test_traj):
    # ref_T is original promp.  dim test_traj = [2,100]

    testx_n=get_normalization(test_traj[0])
    testy_n=get_normalization(test_traj[1])              
    test_T=np.array([testx_n,testy_n]).T

    f_dist= frdist(ref_T,test_T)
    return f_dist


# def get_dtw_cost(ref_T,test_traj):
#     # ref_T is original promp.  dim test_traj = [2,100]

#     testx_n=get_normalization(test_traj[0])
#     testy_n=get_normalization(test_traj[1])              
#     test_T=np.array([testx_n,testy_n]).T
#     alignment = dtw(ref_T,test_T,step_pattern = 'symmetric2' ,distance_only=True)
#     # dtw_c= alignment.distance
#     dtw_c= alignment.normalizedDistance
#     return dtw_c

def direct_f_dist(ref_T,test_traj):
    # ref_T is original promp.  dim test_traj = [2,100]

        
    test_T=test_traj.T

    f_dist= frdist(ref_T,test_T)
    return f_dist




# distance between a point and a rectangle
#  js code:
# function distance(rect, p) {
#   var dx = Math.max(rect.min.x - p.x, 0, p.x - rect.max.x);
#   var dy = Math.max(rect.min.y - p.y, 0, p.y - rect.max.y);
#   return Math.sqrt(dx*dx + dy*dy);
# }
def rect_dist(p,rect):
    _dx = max([rect[0,0]-p[0],0,p[0]-rect[0,1]])
    _dy = max([rect[1,0]-p[1],0,p[1]-rect[1,1]])  
    d=math.sqrt( _dx**2+_dy**2 )

    return d

# distance between a trajectory and a rectangle    

def traj_rect_dist(traj,rect,k):
    dist=[]
    for _p in traj:
        p_dist = rect_dist(_p,rect)
        dist.append(p_dist)
        rew = min(dist)
        # if rew ==0:
        #     rew = -100
    return  rew


def point_dis(a,b):
    return math.sqrt( (a[0]-b[0])**2+(a[1]-b[1])**2)


def curve_dist(ref,traj):
    dist=[]
    cur_len = traj.shape[0]
    cur_ind= np.array([int(cur_len*0.25),int(cur_len*0.50),int(cur_len*0.75)])
    for i in cur_ind:
        _dist = point_dis(ref[i],traj[i])
        dist.append(_dist)
    l =sum(dist)
    return l


def get_para_array(model):
    para=np.zeros(114)
    para[:16] = model.state_dict()['actor.0.weight'].reshape(-1).numpy()
    para[16:24] = model.state_dict()['actor.0.bias'].reshape(-1).numpy()
    para[24:88] = model.state_dict()['actor.2.weight'].reshape(-1).numpy()
    para[88:96] = model.state_dict()['actor.2.bias'].reshape(-1).numpy()
    para[96:112] = model.state_dict()['actor.4.weight'].reshape(-1).numpy()
    para[112:114] = model.state_dict()['actor.4.bias'].reshape(-1).numpy()
    
    return para

def update_para_tensor(model,para_array):
    tensor_list=[]
    tensor_list.append(torch.from_numpy(para_array[:16].reshape([8,2])))
    tensor_list.append(torch.from_numpy(para_array[16:24]))
    tensor_list.append(torch.from_numpy(para_array[24:88].reshape([8,8])))
    tensor_list.append(torch.from_numpy(para_array[88:96]))
    tensor_list.append(torch.from_numpy(para_array[96:112].reshape([2,8])))
    tensor_list.append(torch.from_numpy(para_array[112:114]))

    model.state_dict()['actor.0.weight'].copy_(tensor_list[0])
    model.state_dict()['actor.0.bias'].copy_(tensor_list[1])
    model.state_dict()['actor.2.weight'].copy_(tensor_list[2])
    model.state_dict()['actor.2.bias'].copy_(tensor_list[3])
    model.state_dict()['actor.4.weight'].copy_(tensor_list[4])
    model.state_dict()['actor.4.bias'].copy_(tensor_list[5])