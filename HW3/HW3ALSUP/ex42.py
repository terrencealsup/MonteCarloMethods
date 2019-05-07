"""
File: ex42.py

Author: Terrence Alsup
Date: March 27, 2019
Monte Carlo Methods HW 3

Run script to produce plots from exercise 42.
"""

import numpy as np
import acor
import IsingSamplers
from matplotlib import pyplot as plt

N = int(1E6) # Length of the chain.
L = 10
beta = 0.05

mags = IsingSamplers.IsingSampler(N,L,beta,'Gibbs',method='random')[1]
magMH = IsingSamplers.IsingSampler(N,L,beta,'Metroplis',method='random')[1]

tau = acor.acor(mags, maxlag=700)[0]
tauMH = acor.acor(magMH, maxlag=700)[0]

print("\n")
print("Estimated IAC of Gibbs Sampler    :   ", tau)
print("Estimated IAC of Metroplis Sampler:   ", tauMH)
print("\n")

autocorr = acor.function(mags, maxt=800)
autocorrMH = acor.function(magMH, maxt=800)
plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(autocorr, c='b', label='Gibbs')
plt.plot(autocorrMH, c='r', label='Metroplis Hastings')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Autocorrelation of Magnetization ($L = 10$, $\\beta=0.05$)')
plt.legend()

plt.show()
