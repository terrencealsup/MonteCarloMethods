'''
OVERVIEW:

This program simulates states from a periodic 2D lattice sigma, in which each
sigma(i,j) = +- 1, using 2 MCMC schemes: Gibbs sampling and the
Metropolis-Hastings algorithm. A nearest neighbor Ising model is assumed. Once
both Markov chains are generated, the integrated autocorrelation times for the
magnetization function ( sum_{i,j}(sigma(i,j)) ) are compared between methods.

PARAMETERS:

The grid is LxL
K is the total length of the Markov chain
beta is the inverse temperature term
B is the external magnetic field
if J = 1, the interactions are ferromagnetic
if J = -1, the interactions are antiferromagnetic
kappa is truncation point of the autocorrelation curve and should be
adjusted if autocorrelation curve G does not tend to 0 by G[kappa]
burn-in is pre-chain truncated before Markov process reaches stationarity

REFERENCES:

https://en.wikipedia.org/wiki/Gibbs_sampling
https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
https://en.wikipedia.org/wiki/Ising_model
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font',family='Times New Roman')

def main():

    global L,K,B,beta,J,kappa,failure,burnin
    L = 15
    K = 100000
    beta = 1.0
    B = 0.0
    J = 1.
    kappa = 10000
    failure = 0
    burnin = 1000


#   simulates sigma using Gibbs sampler and MH algorithm
    print "starting Gibbs sampler"
    sigma_Gibbs = Gibbs()
    print "starting Metropolis-Hastings algorithm"
    sigma_MH = MetHast()

#   plots the initial and final states for both sample methods
    plot_state(sigma_Gibbs,sigma_MH)

#   trims off burn-in
    sigma_Gibbs = sigma_Gibbs[:,:,burnin:]
    sigma_MH = sigma_MH[:,:,burnin:]

#   calculates magnetization
    M_Gibbs = magnetization(sigma_Gibbs)
    M_MH = magnetization(sigma_MH)

#   plots the magnetization
    plt.hist(M_Gibbs,bins=15,histtype='step',label='Gibbs: mean(M) = {:.4f}'.\
             format(np.mean(M_Gibbs)))
    plt.hist(M_MH,bins=15,histtype='step',label='MetHast: mean(M) = {:.4f}'.\
             format(np.mean(M_MH)))
    plt.xlabel(r"$M/L^{2}$"); plt.ylabel('occurences')
    plt.legend(fontsize='13'); plt.tight_layout()
    #plt.show()
    plt.savefig('project_plot2.png')
    plt.close()

#   compares the integrated autocorrelation times for magnetizations
    compare_IAC([M_Gibbs,M_MH],labels=['Gibbs','MetHast'])


'''
Each iteration, the Gibbs sampler selects one of the L^2 lattice elements
randomly, i.e. sigma(i,j). A new value of sigma(i,j) is then drawn from
the posterior distribution P[sigma(i,j) | all sigma(I!=i, J!=j)].
The posterior distribution includes only 4 sigma terms because the Ising
model assumes nearest neighbor interactions: sigma(I=i+-1, J=j+-1). Note
that sigma being updated one (i,j) pair at a time is the characteristic
partial resampling feature of Gibbs sampling.
'''
def Gibbs():
#   sigma_series stores sigma for each step of the Markov chain
    sigma_series = initialize_sigma()
#   selects (i,j) randomly and calls draw_sigma_ij(...) to update sigma(i,j)
    for k in range(1,K):
        i,j = np.random.randint(0,L,size=2)
        sigma_series[:,:,k] = Gibbs_update(sigma_series[:,:,k-1],i,j)
    return sigma_series

'''
Returns the new state of sigma after updating sigma(i,j) according to the
posterior distribution used in Gibbs sampling.
'''
def Gibbs_update(sigma,i,j):
#   The nearest neighbors of sigma(i,j) = sigma(i+1,j), sigma(i-1,j),
#   sigma(i,j+1), and sigma(i,j-1).
    neighbors = get_neighbors(sigma,i,j)
#   p and q are the probabilities of being spin +1 or spin -1, respectively.
#   p + q = 1.
    p,q = posterior(sigma[i,j],neighbors,sigma)
#   sets sigma(i,j) according to wp and wm
    if np.random.rand() < p:
        sigma[i,j] = 1
    else:
        sigma[i,j] = -1
#   returns new state for sigma
    return sigma

'''
Each iteration, the Metropolis-Hastings sampler selects one of the L^2
elements randomly, i.e. sigma(i,j). The proposed state for sigma(i,j,k=i+1)
is set to omega(i,j) = -sigma(i,j,k=i). Afterwards, this proposed state is
either accepted or rejected with a probability A.
'''
def MetHast():
#   sigma_series stores sigma for each step of the Markov chain
    sigma_series = initialize_sigma()
#   selects (i,j) randomly and calls draw_sigma_ij(...) to update sigma(i,j)
    for k in range(1,K):
        i,j = np.random.randint(0,L,size=2)
        sigma_series[:,:,k] = MetHast_update(sigma_series[:,:,k-1],i,j)
    return sigma_series

'''
Returns the new state of sigma after updating sigma(i,j) according to the
accept/reject scheme used in the Metropolis Hastings algorithm.
'''
def MetHast_update(sigma,i,j):
#   omega is the proposed state for sigma(i,j)
    omega = -sigma[i,j]
#   The nearest neighbors of sigma(i,j) = sigma(i+1,j), sigma(i-1,j),
#   sigma(i,j+1), and sigma(i,j-1).
    neighbors = get_neighbors(sigma,i,j)
#   sigma(i,j) is either set to sigma or -sigma (the original state)
    sigma[i,j] = acceptreject(omega,neighbors,sigma)
    return sigma

'''
Returns the state for sigma(i,j,k=i+1), after either accepting or rejecting
the proposal, omega.
'''
def acceptreject(omega,neighbors,sigma):
    global failure
#   H_after and H_before are the Hamiltonians for sigma_ij after and before
#   flipping sigma_ij.
    H_after = -omega * J * np.sum(neighbors) - B * np.sum(sigma)
    H_before = omega * J * np.sum(neighbors) - B * np.sum(sigma)
    delta_H = H_after - H_before
#   A is the ratio of Pr(omega)/Pr(sigma) = Pr(H_after)/Pr(H_before)
    A = np.exp(-beta*delta_H)
#   if delta_H < 0, accept transition with 100% probability,
#   otherwise accept with probability A = exp(-beta*delta_H)
    if np.random.rand() < A:
        return omega
    else:
        failure += 1
        return -omega

'''
Returns p = Pr{sigma(i,j) = +1} and q = Pr{sigma(i,j) = -1}. These
are otherwise known as the Boltzmann factors.
'''
def posterior(sigma_ij,neighbors,sigma):
#   Hp and Hm are the Hamiltonians for sigma_ij = +/- 1
    Hp = -1 * J * np.sum(neighbors) - B * np.sum(sigma)
    Hm = +1 * J * np.sum(neighbors) - B * np.sum(sigma)
#   p and q are the probability that sigma_ij is +/- 1
    p = np.exp(-beta*Hp)/(np.exp(-beta*Hp) + np.exp(-beta*Hm))
    q = np.exp(-beta*Hm)/(np.exp(-beta*Hp) + np.exp(-beta*Hm))
    return p,q

'''
Returns the nearest neighbors of sigma(i,j). For nearest neighbor interactions,
these are sigma(i+1,j), sigma(i-1,j), sigma(i,j+1), and sigma(i,j-1).
'''
def get_neighbors(sigma,i,j):
    return np.array([sigma[(i+1)%L,j],
                     sigma[(i-1)%L,j],
                     sigma[i,(j+1)%L],
                     sigma[i,(j-1)%L]])

'''
Returns M, the magnetization. M(sigma) is the function for which the integrated
autocorrelation time is calculated for (equivalent to f(x) in the vignette).
'''
def magnetization(sigma_series):
#   M is the magnetization
    M = np.sum(sigma_series,axis=(0,1))/L**2
    return M

'''
Returns a zero-filled array sigma of dimension (i,j,K) which the Markov chain
is stored in. That's not completely true. Before returning the array, the
initial Markov state is defined in sigma[:,:,0], where each sigma(i,j,k=0) =
+/- 1 with equal probability.
'''
def initialize_sigma():
#   creates sigma(k=0), with elements +/-1 with equal probability
#   sigma_series stores sigma for every time-step
    sigma0 = np.random.choice([-1,1],size=L**2,p=[0.5,0.5]).reshape(L,L)
    sigma_series = np.zeros((L,L,K))
    sigma_series[:,:,0] = sigma0
    return sigma_series

'''
Calculates the autocorrelation curve of M, then returns the integrated
autocorrelation time (IAC).
'''
def tau(M):
#   autocorr is the UNintegrated autocorrelation curve
    autocorr = auto_corr_fast(M)
#   tau = 1 + 2*sum(G)
    return 1 + 2*np.sum(autocorr), autocorr

'''
This is the intuitive and naive way to calculate autocorrelations. See
auto_corr_fast(...) instead.
'''
def auto_corr(M):
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    auto_corr = np.zeros(kappa-1)
    mu = np.mean(M)
    for s in range(1,kappa-1):
        auto_corr[s] = np.mean( (M[:-s]-mu) * (M[s:]-mu) ) / np.var(M)
    return auto_corr

'''
The bruteforce way to calculate autocorrelation curves is defined in
auto_corr(M). The correlation is computed for an array against itself, and
then the indices of one copy of the array are shifted by one and the
procedure is repeated. This is a typical "convolution-style" approach.
An incredibly faster method is to Fourier transform the array first, since
convolutions in Fourier space is simple multiplications. This is the approach
in auto_corr_fast(...)
'''
def auto_corr_fast(M):
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)
#   G is the autocorrelation curve
    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G

'''
Plots the autocorrelation curves and calculates the IAC for each M in Ms
'''
def compare_IAC(Ms,labels):
#   loop through each magnetization chain
    for ind,M in enumerate(Ms):
#       get IAC and autocorrelation curve
        IAC,G = tau(M)
        plt.plot(np.arange(len(G)),G,label="{}: IAC = {:.2f}".\
                                            format(labels[ind],IAC))
    plt.legend(loc='best',fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig('project_plot3.png')
    plt.close()

'''
Plots the initial and final states for both sample methods
'''
def plot_state(sigma1,sigma2):
#   plots sigma(k=0) and sigma(k=K) for sweep and random
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(221)
    ax.pcolormesh(sigma1[:,:,0]); ax.set_title('Gibbs: $k=0$')
    ax = fig.add_subplot(222)
    ax.pcolormesh(sigma1[:,:,-1]); ax.set_title('Gibbs: $k=K$')
    ax = fig.add_subplot(223)
    ax.pcolormesh(sigma2[:,:,0]); ax.set_title('MH: $k=0$')
    ax = fig.add_subplot(224)
    ax.pcolormesh(sigma2[:,:,-1]); ax.set_title('MH: $k=K$')
    plt.tight_layout()
    #plt.show()
    plt.savefig('project_plot1.png')
    plt.close()

main()
