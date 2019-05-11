import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import Rosenbrock

def EnsembleSampler(X0, N, L, alpha):
    # Functions to evaluate the density, grad log density, and Hessian of the
    # minus log density.
    log_pi = Rosenbrock.log_pi

    MCMC_chain = np.zeros((N+1, L, len(X0[0,:])))
    MCMC_chain[0,:,:] = X0 # Initialize the L walkers.

    X = X0
    accept = 0
    for k in range(1, N+1):
        for i in range(L):
            # Update the l-th walker.

            # Randomly select an index j != i
            j =  np.mod(i + 1 + np.random.randint(L-1), L)

            # Inverse transform sampling for Z.
            sa = np.sqrt(alpha)
            Z = ( (sa - 1/sa) * np.random.rand() + 1/sa )**2

            # Get the proposal.
            y = X[j,:] + Z*(X[i,:] - X[j,:])

            d = len(X[0,:]) # Get the dimension.
            # Compute the log of the acceptance probability.
            log_pacc = (d-1)*np.log(Z) + log_pi(y) - log_pi(X[i,:])

            # Metropolize.
            if np.log(np.random.rand()) <= log_pacc:
                accept += 1
                MCMC_chain[k,i,:] = y
            else:
                MCMC_chain[k,i,:] = X[i,:]

        X = MCMC_chain[k,:,:]

    return [MCMC_chain, accept/(N*L)]



def plot_trajectory(X0, N, L, alpha):
    [chain, accept] = EnsembleSampler(X0, N, L, alpha)

    print("\nAcceptance Rate = {:0.1f}%.\n".format(100*accept))

    pi = Rosenbrock.pi
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-2, 10, 100)

    X1, X2 = np.meshgrid(x1, x2, sparse=True)

    Y = pi([X1, X2])

    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.contourf(x1, x2, Y)
    for l in range(L):
        plt.plot(chain[N,l,0], chain[N,l,1], 'r x')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Affine Invariant Ensemble Trajectories')
    plt.show()

L = 50
X0 = 0.5*np.random.randn(2*L)
X0 = X0.reshape((L, 2))
N = 100
alpha = 1.2
plot_trajectory(X0, N, L, alpha)
