import intprim
from intprim.probabilistic_movement_primitives import *
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
from cem import CEM
from utils import *
import time
from pytorch_fid.fid_score import calculate_frechet_distance
from ppo_via import  Agent
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from torch.distributions.multivariate_normal import MultivariateNormal
def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    # parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=5, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=4, type=int, help='update number')
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    parser.add_argument('-batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dim')
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    args = parser.parse_args()
    return args

cfg = get_args()

n_states=2
n_actions=3
agent = Agent(n_states, n_actions, cfg)