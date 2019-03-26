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
        if method == 'random' or sampler != 'Gibbs': # Randomly and independently select a site.
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


def makeArray(samples, n):
    N = np.int(np.sum(n))
    X = np.zeros(samples.shape)
    index = 0
    for i in range(len(n)):
        X[index:index+n[i]] = samples[i]
        index += n[i]
    return X


"""
Resampling to minimize the conditional variance and keep N fixed.
"""
def resample(samples, W):
    N = len(samples)

    n = np.zeros(N)

    U = np.random.rand()
    Wsum = np.cumsum(W)
    for k in range(N):
        Uj = (np.arange(1,N+1)-U)/N
        if k == 0:
            indicator = (Uj < Wsum[k]).astype(int)
        else:
            indicator = np.logical_and(Uj < Wsum[k], Uj >= Wsum[k-1]).astype(int)
        n[k] = np.sum(indicator)

    return makeArray(samples, n.astype(int))


def Jarzynski(N, beta, L, M, nsteps = 100, resampling=False):
    samples = 2*np.random.randint(0,2,(M,L,L))-1 # M samples from pi_0
    W = np.ones(M)/N # The weights of each sample.
    w = np.zeros(M)

    if resampling:
        samples = resample(samples, W)

    for k in range(N):

        # Update the samples and their weights.
        for j in range(M):
            w[j] = computeUnnormalizedPi(L, beta/N, samples[j])

            for n in range(nsteps):
                site = np.random.randint(0,L,2)
                # Draw new samples.
                samples[j] = GibbsUpdate(site, samples[j], beta*k/N, L)

        # Update weights.
        W = W*w/np.dot(W,w)

        if resampling:
            samples = resample(samples, W)

    if resampling:
        return [samples, np.ones(M)/M]
    else:
        return [samples, W]

"""
L = 10
beta = 1.
chainLength = int(2E2)
N = 100
M =  1000

[X, W] = Jarzynski(N, beta, L, M, resampling=False)

mag = np.zeros(M)
mag2 = np.zeros(M)

for j in range(M):
    m = magnetization(X[j])
    mag[j] = m*W[j]
    mag2[j] = m**2*W[j]

print(np.sum(mag))
print(np.sum(mag2))
"""


"""
###-------------------------------------------------------------
###
### Start the main script here.
### Run all the parts for the assignment.
###
###-------------------------------------------------------------
"""

L = 10 # Fix the lattice size.

Nsamples = 1000
chainLength = int(1E4)
beta = .05

mags = np.zeros(Nsamples)

for j in range(Nsamples):
    mags[j] = magnetization(IsingSampler(chainLength, L, beta, sampler='Gibbs',getMagnetization=True)[0])/(L**2)

plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.hist(mags, bins=50, density=True)
plt.xlabel('Average Magnetization $f(\sigma)/L^2$')
plt.ylabel('Relative Frequency')
plt.title('Histogram of the Average Magnetization for $\\beta = 0.05$, $L=10$')
plt.show()





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
