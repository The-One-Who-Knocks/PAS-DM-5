# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt

N = 1000
mu = 5.
Z = mu*(np.exp(-1./mu) - np.exp(-20./mu))

x = np.linspace(1, 20, num=N, dtype=float)
f = np.exp(-x/mu)/Z
F = mu * (np.exp(-1./mu) - np.exp(-x/mu)) / Z

plt.xlim([1,20])
plt.plot(x, f, 'blue', linewidth=2)
plt.savefig("pdf.pdf")
plt.show()
plt.xlim([1,20])
plt.plot(x, F, 'red', linewidth = 2)
plt.savefig("cdf.pdf")
plt.show()
    