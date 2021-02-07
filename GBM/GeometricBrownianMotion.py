import numpy as np
import matplotlib.pyplot as plt
from random import gauss

class GBM:
    
    def __init__(self, num_seeds, isTraining=True):
        self.num_seeds = num_seeds
        self.seed = 0
        self.S0s = np.linspace(50, 200, self.num_seeds)
        if isTraining:
            np.random.seed(1) 
        else:
            np.random.seed(self.num_seeds) 
        np.random.shuffle(self.S0s)
        self.S0 = self.S0s[self.seed]

    def Brownian(self, seed, N):
                          
        dt = 1./252                                    # time step
        b = np.random.normal(0., 1., int(N))*np.sqrt(dt)  # brownian increments
        W = np.cumsum(b)                             # brownian path
        return W, b
   
    def GBM(self, mu, sigma, W, N):
        t = np.arange(0,N+1)/252.0
        S = []
        S.append(self.S0)
        for i in range(1,int(N)):
            drift = (mu - 0.5 * sigma**2) * t[i]
            diffusion = sigma * W[i-1]
            S_temp = self.S0*np.exp(drift + diffusion)
            S.append(S_temp)
        self.seed = (self.seed + 1) % self.num_seeds
        self.S0 = self.S0s[self.seed]
        return S, t

    def generate_stock_paths(self, mu, sigma, N, num_seeds=20000, isTraining=True):
        if isTraining:
            seeds = range(1, num_seeds+1, 1)
            np.random.seed(1) 
        else:
            seeds = range(num_seeds+1, num_seeds*2+1, 1)
            np.random.seed(self.num_seeds) 
        Brownian_motions = [self.Brownian(seed, N)[0] for seed in seeds]
        stocks = [self.GBM(mu, sigma, W, N)[0] for W in Brownian_motions]

        t = np.linspace(0.,1.,N+1)
        np.random.shuffle(stocks)

        return stocks, t

