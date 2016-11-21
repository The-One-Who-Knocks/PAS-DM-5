# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt

N = 1000000
mu = 5.


Z = mu*(np.exp(-1./mu) - np.exp(-20./mu))
x = np.linspace(1, 20, num=N, dtype=float)

f = np.exp(-x/mu)/Z

def sampling_function(a):
    y = np.random.uniform()    
    return -a*np.log(np.exp(-1./a) - Z/a*y)

sample = np.zeros(N)
for i in range(N):
    sample[i] = sampling_function(mu)

plt.xlim([1,20])
n, bins, patches = plt.hist(sample, 50, normed=True, facecolor='green', alpha=0.75)
plt.plot(x, f, 'blue', linewidth = 2)
plt.savefig("distribution.pdf")
plt.show()  