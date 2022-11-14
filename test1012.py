import numpy as np
from math import cos, pi
import matplotlib.pyplot as plt
import random
from random import randint
from tqdm import tqdm
import os



class NES:
    def __init__(self):
        super().__init__()

        # Initialize mean and standard deviation
        #self.theta_mean = np.zeros((n_samples, dim_theta))
        self.theta_mean  = np.random.uniform(min_val, max_val, (n_samples, dim_theta))
        self.theta_std = np.random.uniform(max_val-1, max_val, (n_samples, dim_theta))
        self.n_samples = n_samples
        self.t = n_iterations
        self.top_p = top_p
        self.fit_gaussian()

    def fit_gaussian(self):
        # theta is actualy the population sampled from the distribution
        self.theta = np.random.normal(self.theta_mean, self.theta_std)
        #self.theta = np.clip(theta, min_val, max_val)

        

    def generation(self, function=0):
        # Sample n_sample candidates from N(theta)
        mean_fitness = []
        best_fitness = []
        worst_fitness = []
        I = np.identity(dim_theta*2)
        for i in tqdm(range(0, self.t)):
            #0 --> Sphere; 1 --> Rasti
            fitness = self.evaluate_fitness(self.theta, function)

            mean_fitness.append(np.mean(fitness))
            best_fitness.append(np.min(fitness))
            worst_fitness .append(np.max(fitness))

            
            # if plot == 1:
            #     self.plot_candidates(self.theta, function, min_val, max_val)
            #     plt.pause(pause)


            # Compute the two gradient separately 
            Dlog_mean = self.compute_mean_grad(self.theta)
            Dlog_std = self.compute_std_grad(self.theta)

            Dlog = np.concatenate((Dlog_mean, Dlog_std), axis=1)

            Dj = np.mean(Dlog * np.array([fitness]).T, axis=0)

            F = np.zeros((Dlog.shape[1], Dlog.shape[1]))
        
            for i in range(Dlog.shape[0]):
                F = F + np.outer(Dlog[i,:], Dlog[i,:])
            
            F = F / self.n_samples
            F = F + I * 1e-5

            theta = np.concatenate((self.theta_mean, self.theta_std), axis=1)

            Theta = theta - alpha * np.dot(np.linalg.inv(F), Dj)

            self.theta_mean = Theta[:, :int(Theta.shape[1]/2)]
            self.theta_std = Theta[:, int(Theta.shape[1]/2):]
            self.fit_gaussian()

            if plot == 1:
                plt.close("all")
        if plot == 1:        
            plt.show()


        print("mean fitness level")
        print(mean_fitness)


        plt.plot(mean_fitness)
        plt.show()

        return mean_fitness, best_fitness, worst_fitness
        
        




    def compute_mean_grad(self, e_candidates):
        eps = 1e-6
        N = e_candidates - self.theta_mean
        D = self.theta_std ** 2
        return N/D

    def compute_std_grad(self, e_candidates):
        eps = 1e-6
        N = (e_candidates - self.theta_mean)**2 - self.theta_std**2
        D = self.theta_std ** 3
        return N/D


    def evaluate_fitness(self, candidates, func=0):
        if func == 0:
            return SphereFunction(candidates)
        else:
            return rastriginFunction(candidates)

        _r = np.zeros([self.N])
        for i in range(self.N):
            _r[i]= self.func(x[i])
        return _r


    # def plot_candidates(self, candidates, func=0, min_val=-5, max_val=5):

    #     if func == 0:
    #         plot_visualize_sphere(1000, candidates, min_val, max_val)
    #     else:
    #         plot_visualize_rastrigin(1000, candidates, min_val, max_val)
