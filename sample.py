# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt

# Simulation parameters
N = 1000000   # Number of sampled points
mu = 5.       # Distribution's parameter
Z = mu*(np.exp(-1./mu) - np.exp(-20./mu)) # Partition function
x = np.linspace(1, 20, num=N, dtype=float) # Domain

f = np.exp(-x/mu)/Z  # PDF

def sampling_function(a):
    """ Function which samples points according to the distribution (1)"""
    y = np.random.uniform()    
    return -a*np.log(np.exp(-1./a) - Z/a*y)

sample = np.zeros(N)  # Set of sampled points
for i in range(N):
    sample[i] = sampling_function(mu)

plt.xlim([1,20])
n, bins, patches = plt.hist(sample, 50, normed=True, facecolor='green', alpha=0.75)
plt.plot(x, f, 'blue', linewidth = 2)
plt.savefig("distribution.pdf")
plt.show()  