"""
File: IsingSamplers.py

Author: Terrence Alsup
Date: March 27, 2019
Monte Carlo Methods HW 3

Implement different MCMC schemes to sample from the 2D Ising Model.
Gibbs Sampler: with randomly selecting sites as well as sweeping
Metroplis-Hastings
Jarzynski: with and without resampling
"""

import numpy as np

"""
Compute the magnetization of a vector of spins.

@param spins: L-by-L matrix of +-1 values
@return : the magnetization of the lattice
"""
def magnetization(spins):
    return np.sum(spins)

"""
Compute the unnormalized distribution of the 2D Ising model.

@param L: the size of the lattice
@param beta: the inverse temperature > 0
@param spins: the L-by-L matrix of +-1 values
"""
def computeUnnormalizedPi(L, beta, spins):
    sum = 0.
    for i in range(L):
        for j in range(L):
            sum += spins[i,j]*(spins[np.mod(i+1,L),j] + spins[np.mod(i-1,L),j] + spins[i,np.mod(j+1,L)] + spins[i,np.mod(j-1,L)])
    return np.exp(beta*sum)

"""
Do one step of Gibbs Sampling.

@param site: the site to update
@param spins: the L-by-L matrix of +-1 spins
@param beta: the inverse temperature > 0
@param L: the size of the lattice
"""
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

"""
Do one step of Metroplis-Hastings.

@param site: the site to update
@param spins: the L-by-L matrix of +-1 spins
@param beta: the inverse temperature > 0
@param L: the size of the lattice
"""
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


"""
Draw a single sample from the Ising model using MCMC.

@param chainLength: the length of the Markov chain
@param L: the size of the lattice
@param beta: the inverse temperature
@param sampler: the sampler to use either Gibbs or Metropolis'
@param method: the method to select the sites, either random or sweep
@param getMagnetization: return the magnetization at each step if true

@return spins: the sample from the Ising model
@return mag: the magnetization at each step
"""
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

"""
Make an array of the samples with n[i] corresponding to the number of times to
take samples[i].

@param samples: the samples to choose from
@param n: the vector of the number of each sample to take

@return X: the vector containing the appropriate number of each sample
"""
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

@param samples: the samples to resample from
@param W: the weights of each sample

@return : the vector or resampled samples with uniform weights
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

"""
Implements Jarzynski's method with or without resampling.

@param N: the number of interpolating distributions
@param beta: the inverse temperature
@param L: the lattice size
@param M: the number of particles to use
@param nsteps: the number of Metropolis steps to use in each transition
@param resampling: do resampling if true

@return samples: M samples from the Ising model
@return W: the weights of the returned samples
"""
def Jarzynski(N, beta, L, M, nsteps = 200, resampling=False):
    samples = 2*np.random.randint(0,2,(M,L,L))-1 # M samples from pi_0
    W = np.ones(M)/N # The weights of each sample.
    w = np.zeros(M)

    # Do the initial resampling.
    if resampling:
        samples = resample(samples, W)

    # Interpolate between pi_0 and pi.
    for k in range(N):

        # Update the samples and their weights.
        for j in range(M):
            w[j] = computeUnnormalizedPi(L, beta/N, samples[j])

            # Transition operator is many Metroplis steps.
            for n in range(nsteps):
                site = np.random.randint(0,L,2)
                # Draw new samples.
                samples[j] = MetroplisUpdate(site, samples[j], beta*k/N, L)

        # Update weights.
        W = W*w/np.dot(W,w)

        # Resample.
        if resampling:
            samples = resample(samples, W)
            W = np.ones(M)/M


        return [samples, W]
