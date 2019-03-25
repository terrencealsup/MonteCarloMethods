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

def computeUnnormalizedPi(L, beta, spins):
    sum = 0.
    for i in range(L):
        for j in range(L):
            sum += spins[i,j]*(spins[np.mod(i+1,L),j] + spins[np.mod(i-1,L),j] + spins[i,np.mod(j+1,L)] + spins[i,np.mod(j-1,L)])
    return np.exp(beta*sum)


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
                site = [np.mod(np.floor_divide(L*i+j+1,L),L), np.mod(L*i+j+1,L)]

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


def Jarzynski(N, beta, L, M, resample=True):
    samples = 2*np.random.randint(0,2,(M,L,L))-1 # M samples from pi_0
    W = np.ones(M) # The weights of each sample.
    w = np.zeros(M)

    for k in range(N):
        # Update the samples and their weights.
        for j in range(M):
            w[j] = computeUnnormalizedPi(L, beta/N, samples[j])
            site = np.random.randint(0,L,2)
            # Draw new samples.
            samples[j] = GibbsUpdate(site, samples[j], beta*k/N, L)

        # Update weights.
        W = W*w/np.dot(W,w)

    return [samples, W]

L = 10
beta = 0.05
chainLength = int(1E3)
M =  100

[X, W] = Jarzynski(chainLength, beta, L, M)

mag = 0.
for j in range(M):
    mag += magnetization(X[j])*W[j]

print(mag)


"""
N = 100
mags = np.zeros(N)
for i in range(N):
    mags[i] = magnetization(IsingSampler(chainLength,L,beta,'Gibbs','sweep',False))

plt.figure()
plt.hist(mags)
"""

"""
mags = IsingSampler(chainLength,L,beta,'Gibbs',method='sweep')[1]



fun = acor.function(mags, maxt=2000)


tau = acor.acor(mags, maxlag=1000)[0]
print(tau)


plt.figure()
plt.plot(fun)

plt.show()
"""
