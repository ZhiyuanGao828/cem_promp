import numpy as np
import pandas as pd
from frechetdist import frdist
import math
# from pytorch_fid.fid_score import calculate_frechet_distance
# from two_dof.toy_obstacle import ToyObstacle
# from dtw import *

#  traj dim=[100,2]
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
    return math.sqrt( _dx**2+_dy**2 )

# distance between a trajectory and a rectangle    

def traj_rect_dist(traj,rect):
    dist=[]
    for _p in traj:
        p_dist = rect_dist(_p,rect)
        dist.append(p_dist)
    
    return min(dist)


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



# def cool_distance(x,y):



        #     toy_obstacle.eval_distance(
        #     np.array([3,4,5,6]),
        #     np.array([3,4,5,6])
        # )