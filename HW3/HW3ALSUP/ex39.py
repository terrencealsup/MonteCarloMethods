"""
File: ex39.py

Author: Terrence Alsup
Date: March 27, 2019
Monte Carlo Methods HW 3

Run script to produce plots from exercise 39.
"""

import numpy as np
import acor
import IsingSamplers
from matplotlib import pyplot as plt

###
### Produce the histograms of the magnetization for beta = 1 and beta = 0.05.
###
L = 10 # Fix the lattice size.
beta = 0.05

Nsamples = 1000
chainLength = int(1E4)

magsb1 = np.zeros(Nsamples)
magsb005 = np.zeros(Nsamples)

for j in range(Nsamples):
    magsb1[j] = magnetization(IsingSampler(chainLength, L, 1.0, sampler='Gibbs',getMagnetization=True)[0])/(L**2)
    magsb005[j] = magnetization(IsingSampler(chainLength, L, 0.05, sampler='Gibbs',getMagnetization=True)[0])/(L**2)

plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.hist(magsb1, bins=50, density=True)
plt.xlabel('Average Magnetization $f(\sigma)/L^2$')
plt.ylabel('Relative Frequency')
plt.title('Histogram of the Average Magnetization for $\\beta = 1$, $L=10$')

plt.figure(2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.hist(magsb005, bins=50, density=True)
plt.xlabel('Average Magnetization $f(\sigma)/L^2$')
plt.ylabel('Relative Frequency')
plt.title('Histogram of the Average Magnetization for $\\beta = 1$, $L=10$')

##
## Plot the autocorrelation function for beta = 0.05 with random and sweeping.
##

N = int(1E7) # Length of the chain.

mags = IsingSamplers.IsingSampler(N,L,beta,'Gibbs',method='random')[1]
magSweep = IsingSamplers.IsingSampler(N,L,beta,'Gibbs',method='sweep')[1]

tau = acor.acor(mags, maxlag=700)[0]
tauSweep = acor.acor(magSweep, maxlag=700)[0]

print("\n")
print("Estimated IAC of Gibbs Sampler (random):   ", tau)
print("Estimated IAC of Gibbs Sampler (sweeping): ", tauSweep)
print("\n")

autocorr = acor.function(mags, maxt=800)
autocorrSweep = acor.function(magSweep, maxt=800)
plt.figure(3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(autocorr, c='b', label='Gibbs')
plt.plot(autocorrSweep, c='r', label='Metroplis Hastings')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Autocorrelation of Magnetization ($L = 10$, $\\beta=0.05$)')
plt.legend()

##
## Now change the temperature for beta = 1.
##

beta = 1.
N = int(1E7)

mags = IsingSamplers.IsingSampler(N,L,beta,'Gibbs',method='random')[1]

autocorr = acor.function(mags, maxt=800)
plt.figure(4)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(autocorr, 'b')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Autocorrelation of Magnetization ($L = 10$, $\\beta=1$)')




##
## Now change the lattice size with beta = 0.05 fixed.
##

Lsizes = [4,6,8,10,12]
beta = 0.05
N = int(1E7)

taus = np.zeros(5)
i = 0
for L in Lsizes:
    mags = IsingSamplers.IsingSampler(N,L,beta,'Gibbs')[1]
    taus[i] = acor.acor(mags, maxlag=700)[0]
    i += 1

plt.figure(5)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(Lsizes, taus, 'b-x')
plt.xlabel('Lattice Size $L$')
plt.ylabel('IAC')
plt.title('IAC vs. $L$ ($\\beta = 0.05$)')



plt.show()
