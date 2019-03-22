import numpy as np
import acor
from matplotlib import pyplot as plt

"""
Compute the magnetization of a vector of spins.

@param spins: (numpy.ndarray) L-by-L matrix of +-1 values.
@return : (float) the magnetization of the lattice
"""
def magnetization(spins):
    return np.sum(spins)


def GibbsUpdate(site, spins, beta, L):
    i = site[0]
    j = site[1]
    sum = spins[np.mod(i+1,L),j] + spins[np.mod(i-1,L),j] + spins[i,np.mod(j+1,L)] + spins[i,np.mod(j-1,L)]

    U = np.random.rand()
    w = np.exp(2*beta*sum)
    if U <= w/(w + 1/w):
        spins[i,j] = 1
    else:
        spins[i,j] = -1

    return spins


def MetroplisUpdate(site, spins, beta, L):
    i = site[0]
    j = site[1]
    sum = spins[np.mod(i+1,L),j] + spins[np.mod(i-1,L),j] + spins[i,np.mod(j+1,L)] + spins[i,np.mod(j-1,L)]
    p_acc = np.exp(-4*beta*spins[i,j]*sum)

    U = np.random.rand()
    if U <= p_acc:
        spins[i,j] = -1*spins[i,j]
        return spins
    else:
        return spins



def IsingSampler(chainLength, L, beta, sampler, method='random', getMagnetization=True):
    spins = 2*np.random.randint(0,2,(L,L))-1 # Generate random initial spins.

    if getMagnetization:
        mag = np.zeros(1+chainLength)
        mag[0] = magnetization(spins)

    for k in range(chainLength):
        if method == 'random': # Randomly and independently select a site.
            site = np.random.randint(0,L,2)
        else: # Sweep through the sites deterministically.
            if k == 0:
                site = [0,0]
            else:
                i = site[0]
                j = site[1]
                site = np.asarray([np.mod(L*i+j+1,L), np.remainder(L*i+j+1,L)])

        # Determine which sampler to use.
        if sampler == 'Gibbs' or method != 'random':
            spins = GibbsUpdate(site, spins, beta, L)
        else:
            spins = MetroplisUpdate(site, spins, beta, L)

        if getMagnetization:
            mag[k+1] = magnetization(spins)

    if getMagnetization:
        return [spins, mag]
    else:
        return spins



L = 15
beta = 1.
chainLength = int(1E5)

mags = IsingSampler(chainLength,L,beta,'Gibbs')[1]



plt.figure()
plt.plot(fun)
plt.show()
