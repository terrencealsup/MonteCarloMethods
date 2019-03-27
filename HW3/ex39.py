import numpy as np
import acor
import IsingSamplers
from matplotlib import pyplot as plt

"""
L = 10 # Fix the lattice size.

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
plt.show()
"""


##
## Plot the autocorrelation function.
##

L = 10
beta = 0.05
N = int(1E7) # Length of the chain.

#mags = IsingSamplers.IsingSampler(N,L,beta,'Gibbs',method='random')[1]
magSweep = IsingSamplers.IsingSampler(N,L,beta,'Metroplis',method='random')[1]

tau = acor.acor(magSweep, maxlag=700)[0]

"""
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
plt.show()
"""


print("Estimated IAC: ", tau)
