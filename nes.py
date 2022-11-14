import numpy as np
from math import cos, pi
import matplotlib.pyplot as plt
import random
from random import randint
from tqdm import tqdm
import os


class NES():
    def __init__(self, func, d, initial_mu=None, initial_sigma=None,alpha = 1e-3,maxits=30,N=100, Ne=25, epsilon=1e-15,argmin=True,v_min=None, v_max=None, init_scale=1):
        self.func = func                  # target function
        self.d = d                        # dimension of function input X
        self.maxits = maxits              # maximum iteration
        self.N = N                        # sample N examples each iteration
        self.Ne = Ne                      # using better Ne examples to update mu and sigma
        self.reverse = not argmin         # try to maximum or minimum the target function
        self.v_min = v_min                # the value minimum
        self.v_max = v_max                # the value maximum
        self.init_coef = init_scale       # sigma initial value
        self.epsilon=epsilon 
        self.alpha =alpha  #learning rate
        self.state_buf=  np.zeros([maxits,2])
        self.reward_buf=  np.zeros(maxits)
        self.init_mu = initial_mu
        self.init_sigma=initial_sigma
    
        self.theta_mean  = np.random.uniform(self.v_min, self.v_max, (self.N, self.d))
        self.theta_std = np.random.uniform(self.v_max-1, self.v_max, (self.N, self.d))
        self.fit_gaussian()


        # theta.shape (100, 2)  reward shape (100,)
    def fit_gaussian(self):
        # theta is actualy the population sampled from the distribution
        self.theta = np.random.normal(self.theta_mean, self.theta_std)
        #self.theta = np.clip(theta, min_val, max_val)

    def generation(self):
        # Sample n_sample candidates from N(theta)
        mean_fitness = []
        best_fitness = []
        worst_fitness = []
        I = np.identity(self.d*2)
        for i in tqdm(range(0, self.maxits)):
            fitness = self.evaluate_fitness(self.theta)
            mean_fitness.append(np.mean(fitness))
            best_fitness.append(np.min(fitness))
            worst_fitness .append(np.max(fitness))

            # Compute the two gradient separately 
            Dlog_mean = self.compute_mean_grad(self.theta)
            Dlog_std = self.compute_std_grad(self.theta)
            Dlog = np.concatenate((Dlog_mean, Dlog_std), axis=1)
            Dj = np.mean(Dlog * np.array([fitness]).T, axis=0)

            F = np.zeros((Dlog.shape[1], Dlog.shape[1]))
        
            for i in range(Dlog.shape[0]):
                F = F + np.outer(Dlog[i,:], Dlog[i,:])
            
            F = F / self.N 
            F = F + I * 1e-5

            theta = np.concatenate((self.theta_mean, self.theta_std), axis=1)

            Theta = theta - self.alpha * np.dot(np.linalg.inv(F), Dj)

            self.theta_mean = Theta[:, :int(Theta.shape[1]/2)]
            self.theta_std = Theta[:, int(Theta.shape[1]/2):]
            self.fit_gaussian()

        return mean_fitness, best_fitness, worst_fitness

    def compute_mean_grad(self, e_candidates):
        # eps = 1e-6
        N = e_candidates - self.theta_mean
        D = self.theta_std ** 2
        return N/D

    def compute_std_grad(self, e_candidates):
        # eps = 1e-6
        N = (e_candidates - self.theta_mean)**2 - self.theta_std**2
        D = self.theta_std ** 3
        return N/D


    def evaluate_fitness(self, candidates):
        _r = np.zeros([self.N])
        for i in range(self.N):
            _r[i]= self.func(candidates[i])
        return _r








