import numpy as np

class CEM():
    def __init__(self, func, mu_0, sigma_0,maxits=30,N=100, Ne=25,argmin=True,v_min=None, v_max=None, fix_time = False, init_scale=1):
        self.func = func                  # target function
        self.d = mu_0.shape[0]                     # dimension of function input X
        self.maxits = maxits              # maximum iteration
        self.N = N                        # sample N examples each iteration
        self.Ne = Ne                      # using better Ne examples to update mu and sigma
        self.reverse = not argmin         # try to maximum or minimum the target function
        self.v_min = v_min                # the value minimum
        self.v_max = v_max                # the value maximum
        self.init_coef = init_scale       # sigma initial value
        # self.epsilon=epsilon 
        self.state_buf=  np.zeros([maxits,self.d])
        self.reward_buf=  np.zeros(maxits)
        # self.rew_buf_0=  np.zeros(maxits)
        # self.rew_buf_1=  np.zeros(maxits)
        # self.rew_buf_2=  np.zeros(maxits)
        self.fix_time = fix_time
        if fix_time == False:
            self.dof = 3
        else:
            self.dof= 2

        self.init_mu = mu_0

        if fix_time == False:
            _sig=np.ones(self.d)*sigma_0
            _sig[2:self.d:self.dof] = 0.3    
        else:
            _sig=np.ones(self.d)*sigma_0

        self.init_sigma=_sig


    def evalGaussian(self):
        # initial parameters
        t, mu, sigma = self.__initGaussianParams()
        # random sample all dimension each time
        while t < self.maxits:
            x = self.__gaussianSampleData(mu, sigma)
            s = self.__functionReward(x).reshape(-1, 1)
            #print(s.shape)
            x = self.__sortSample(s,x)
              # update parameters
            mu, sigma = self.__updateGaussianParams(x)
            self.state_buf[t,:] = mu 
            self.reward_buf[t] = self.func(mu)
            print('iteration {ite}, v={rew}'.format(ite=t,rew=self.reward_buf[t])) 
            t += 1
        return mu

    def evalGaussian_fixedT(self):
        # initial parameters
        t, mu, sigma = self.__initGaussianParams()
        # random sample all dimension each time
        while t < self.maxits:
            x = self.__gaussianSampleData_fixedT(mu, sigma)
            s = self.__functionReward(x).reshape(-1, 1)
            #print(s.shape)
            x = self.__sortSample(s,x)
              # update parameters
            mu, sigma = self.__updateGaussianParams(x)
            self.state_buf[t,:] = mu 
            self.reward_buf[t] = self.func(mu)
            print('iteration {ite}, v={rew}'.format(ite=t,rew=self.reward_buf[t])) 
            t += 1
        return mu



    def __initGaussianParams(self):
        #initial parameters t, mu, sigma  /t is interation
        t = 0
        mu= self.init_mu
        sigma = self.init_sigma
        return t, mu, sigma

    def __updateGaussianParams(self, x):
        # update parameters mu, sigma
        mu = x[0:self.Ne,:].mean(axis=0)
        sigma = x[0:self.Ne,:].std(axis=0)
        return mu, sigma
    
    def __gaussianSampleData(self, mu, sigma):
        # sample N examples
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j], size=(self.N,))
        if self.v_min is not None and self.v_max is not None:
            sample_matrix[:,0:self.d:self.dof] = np.clip(sample_matrix[:,0:self.d:self.dof],  self.v_min[0], self.v_max[0])
            sample_matrix[:,1:self.d:self.dof] = np.clip(sample_matrix[:,1:self.d:self.dof],  self.v_min[1], self.v_max[1])
            sample_matrix[:,2:self.d:self.dof] = np.clip(sample_matrix[:,2:self.d:self.dof],  self.v_min[-1], self.v_max[-1])
        return sample_matrix

    def __gaussianSampleData_fixedT(self, mu, sigma):
        # sample N examples
        sample_matrix = np.zeros((self.N, self.d))
        for j in range(self.d):
            sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j], size=(self.N,))
            # sample_matrix[:,2:self.d:self.dof] = mu[2:self.d:self.dof]
#         if self.v_min is not None and self.v_max is not None:
#             sample_matrix[:,2:self.d:self.dof] = np.clip(sample_matrix[:,2:self.d:self.dof],  self.v_min[-1], self.v_max[-1])
        return sample_matrix 

    # def evalUniform(self):
    #     # initial parameters
    #     t, _min, _max = self.__initUniformParams()
    #     # random sample all dimension each time
    #     while t < self.maxits:
    #         x = self.__uniformSampleData(_min, _max)
    #         s = self.__functionReward(x).reshape(-1, 1)
    #         #print(s.shape)
    #         x = self.__sortSample(s,x)
    #           # update parameters
    #         _min, _max = self.__updateUniformParams(x)
    #         _mean=(_min + _max) / 2.
    #         self.state_buf[t,:] = _mean 
    #         self.reward_buf[t] = self.func(_mean)
    #         print('iteration {ite}, v={rew}'.format(ite=t,rew=self.reward_buf[t])) 
    #         t += 1
    #     return _mean

    # def __initUniformParams(self):
    #     t = 0
    #     _min = self.v_min if self.v_min else -np.ones(self.d)
    #     _max = self.v_max if self.v_max else  np.ones(self.d)
    #     return t, _min, _max

    # def __updateUniformParams(self, x):
    #     _min = np.amin(x[0:self.Ne,:], axis=0)
    #     _max = np.amax(x[0:self.Ne,:], axis=0)
    #     return _min, _max

    # def __uniformSampleData(self, _min, _max):
    #     sample_matrix = np.zeros((self.N, self.d))
    #     for j in range(self.d):
    #         sample_matrix[:,j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
    #     return sample_matrix

    def __functionReward(self,  x):
        # x [100,2] 
        _r = np.zeros([self.N])
        for i in range(self.N):
            _r[i]= self.func(x[i])
        return _r

    def __sortSample(self, s, x):
    #   sort data by function return
        y=np.concatenate((s,x),axis=1).tolist()
        y = sorted(y, key=lambda x: x[0])
        y=np.array(y)
        x =y[:,1:]
        return x