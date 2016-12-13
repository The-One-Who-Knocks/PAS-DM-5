# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import scipy.optimize
import scipy.misc

# Definition of several functions
def partition_function(mu):
    return mu*(np.exp(-1./mu) - np.exp(-20./mu))
    
def trunc_exponential(x, mu):
    return np.exp(-x/mu)/partition_function(mu)    
    
def random_trunc_exponential(mu):
    """ Function which samples points according to the truncated exponential distribution"""
    y = np.random.uniform()    
    return -mu*np.log(np.exp(-1./mu) - partition_function(mu)*y/mu)

def log_likelihood(mu, sample):
    log_likelihood = 0.
    for i in range(len(sample)):
        log_likelihood -= np.log(trunc_exponential(sample[i], mu))
    return log_likelihood

def sampling_points(N, mu):
    sample = np.zeros(N)
    for i in range(N):
        sample[i] = random_trunc_exponential(mu)
    return sample
    
def mle_estimator(sample):
    return scipy.optimize.minimize(log_likelihood, 1., args=(sample), method='Nelder-Mead').x[0]

        
n_points = 10000
n_sample = 10
mu = np.linspace(0.1, 15, num = 50, dtype=float)

mse = np.zeros(len(mu))
for i in range(len(mu)):
    for j in range(n_sample):
        sample = sampling_points(n_points, mu[i])
        mse[i] += (mu[i] - mle_estimator(sample))**2
    mse[i] /= n_sample

#    
#for j in range(len(mu_domain)):
#    for i in range(n_sample):
#        sample = sampling_points(n_points, mu_domain[j])
#        likelihood_function = likelihood(sample, mu_domain)
