# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import operator


# Definition of several functions
def partition_function(mu):
    return mu*(np.exp(-1./mu) - np.exp(-20./mu))
    
def trunc_exponential(x, mu):
    return np.exp(-x/mu)/partition_function(mu)    
    
def random_trunc_exponential(mu):
    """ Function which samples points according to the truncated exponential distribution"""
    y = np.random.uniform()    
    return -mu*np.log(np.exp(-1./mu) - partition_function(mu)*y/mu)

def likelihood(sample, mu_domain):
    likelihood_function = np.ones(len(mu_domain), dtype=float)
    for i in range(len(mu_domain)):
        for j in range(len(sample)):
            likelihood_function[i] *= trunc_exponential(sample[j], mu_domain[i])*10
            # The factor of 10 in previous line prevents the likelihood to be null because of precision
            # As we normalize the likelihood it doesn't matter, but if we don't, we should drop it
            # If we don't want to use this trick we just have to take le log-likelihood
    return likelihood_function

def log_likelihood(sample, mu_domain):
    log_likelihood_function = np.zeros(len(mu_domain), dtype=float)
    for i in range(len(mu_domain)):
        for j in range(len(sample)):
            log_likelihood_function[i] += np.log(trunc_exponential(sample[j], mu_domain[i]))
    return log_likelihood_function

def sampling_points(N, mu):
    sample = np.zeros(N)
    for i in range(N):
        sample[i] = random_trunc_exponential(mu)
    return sample
    
def normalize(likelihood_function):
    max_value = max(enumerate(likelihood_function), key=operator.itemgetter(1))[1]
    return likelihood_function/max_value

def arg_max(likelihood_function, mu_domain):
    max_index = max(enumerate(likelihood_function), key=operator.itemgetter(1))[0]
    return mu_domain[max_index]
        
# Simulation parameters
n_plot = 100   # Number of iterations in mu for the likelihood
mu = 5.        # Distribution's parameter
mu_domain = np.linspace(0.1, 15, num = n_plot, dtype=float)


#sample = sampling_points(10, mu)
#likelihood_function = likelihood(sample, mu_domain)
#print(arg_max(likelihood_function, mu_domain))
#likelihood_function = normalize(likelihood_function)
#plt.plot(mu_domain, likelihood_function, label="n = 10")
#
#
#sample = sampling_points(100, mu)
#likelihood_function = likelihood(sample, mu_domain)
#print(arg_max(likelihood_function, mu_domain))
#likelihood_function = normalize(likelihood_function)
#plt.plot(mu_domain, likelihood_function, label="n = 100")
#
#
#sample = sampling_points(1000, mu)
#likelihood_function = likelihood(sample, mu_domain)
#print(arg_max(likelihood_function, mu_domain))
#likelihood_function = normalize(likelihood_function)
#plt.plot(mu_domain, likelihood_function, label="n = 1000")

sample = sampling_points(100, 1.)
likelihood_function = likelihood(sample, mu_domain)
print(arg_max(likelihood_function, mu_domain))
likelihood_function = normalize(likelihood_function)
plt.plot(mu_domain, likelihood_function, label="lambda = 1")


sample = sampling_points(100, 5.)
likelihood_function = likelihood(sample, mu_domain)
print(arg_max(likelihood_function, mu_domain))
likelihood_function = normalize(likelihood_function)
plt.plot(mu_domain, likelihood_function, label="lambda = 5")


sample = sampling_points(100, 10.)
likelihood_function = likelihood(sample, mu_domain)
print(arg_max(likelihood_function, mu_domain))
likelihood_function = normalize(likelihood_function)
plt.plot(mu_domain, likelihood_function, label="lambda = 10")

plt.legend()
plt.savefig("likelihood_var_mu.pdf")
plt.show()