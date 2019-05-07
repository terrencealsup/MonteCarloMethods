"""
File: ex61_alsup.py
Author: Terrence Alsup
Date: April 17, 2019
Monte Carlo Methods HW 4

Sample from the XY model using a metropolized and un-metropolized Hybrid Monte
Carlo.  Plot 2 samples at different temperatures.  Compute the IACs for
the cosine of the angle of the magnetization vector.
Does Exercise 61 in the notes.
"""
import XYmodel
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
import acor

def logp_acc(xy, Xh, Xt, Yh, Yt):
    """
    Compute the logarithm of the acceptance probability.

    [Xh, Xt] is the current location of the chain.
    [Yh, Yt] is the new proposal.
    """
    log_piy = -xy.log_density(Yh) + 0.5*linalg.norm(Yt)**2
    log_pix = -xy.log_density(Xh) + 0.5*linalg.norm(Xt)**2

    return log_piy - log_pix

def velocityVerlet(XY, yh, yt, h, n):
    """
    Do n steps of velocity verlet.

    grad K(x) = x since K(x) = |x|^2/2
    """
    for l in range(n):
        yt_temp = yt + (0.5 * h * XY.grad_log_density(yh))
        yhp1 = yh + (h * yt_temp)
        ytp1 = yt_temp + (0.5 * h * XY.grad_log_density(yhp1))
        yh = yhp1
        yt = ytp1

    return [yhp1, ytp1]


def HybridMC(L, beta, h, n, Nsteps, metropolize=False, getMags=False):
    """
    Generate a sample from the XY model using Hybrid Monte Carlo.

    K(x) = |x|^2/2 so Y~N(0, I)
    \hat{J} = I
    """
    xy = XYmodel.XYmodel(L, beta)

    if getMags:
        mag = np.zeros(Nsteps+1)
        mag[0] = xy.cosMagnetVector()

    if metropolize:
        rejected = 0


    # Do Nsteps of MCMC.
    for k in range(1, Nsteps + 1):

        yt_old = np.random.randn(L)   # \tilde{Y}^{(k-1)} drawn from exp(-K(x))

        # Run the Hamiltonian dynamics using velocity verlet for n steps.
        [yh, yt] = velocityVerlet(xy, xy.theta, yt_old, h, n)

        if metropolize:
            # Compute the acceptance probability.
            if np.log(np.random.rand()) < logp_acc(xy, xy.theta, yt_old, yh, yt):
                # Note that we do not need to reset the momentum variables
                # because we sample them at each step independently.
                xy.set(yh)
            else:
                rejected += 1
        else:
            xy.set(yh)

        if getMags:
            mag[k] = xy.cosMagnetVector()

    if metropolize:
        rej_rate = float(rejected)/Nsteps

    if getMags and metropolize:
        return [xy, rej_rate, mag]
    elif getMags and not metropolize:
        return [xy, mag]
    elif not getMags and metropolize:
        return [xy, rej_rate]
    else:
        return xy


def test_sampler():
    """
    Test the un-metropolized XY model sampler.
    """
    L = 25              # Lattice size
    beta = 0.1          # Inverse temperature
    h = 1E-2           # Step size
    n = 10              # Number of velocity verlet steps.
    Nsteps = int(1E4)   # Number of MCMC steps

    # First plot the spins on the circle for large beta.

    # Get points on the circle.
    angle = np.linspace(0, 2*np.pi, 1000)
    circle = [np.cos(angle), np.sin(angle)]

    # Sample from the model and get the spin vectors.
    xy = HybridMC(L, beta, h, n, Nsteps)
    spins = xy.spinVectors()

    # Now plot the result.
    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("XY model sample ($L = {:d}$ and $\\beta = $ {:0.1f})".format(L, beta))
    plt.xlabel('$\\vec{\\sigma}_x$')
    plt.ylabel('$\\vec{\\sigma}_y$')
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot(spins[:,0], spins[:,1], 'bo')
    plt.plot(circle[0], circle[1], 'k:')

def test_IAC():
    """
    Test the un-metropolized XY model sampler and get the ACF and IAC.

    Nsteps is the number of MCMC steps.
    """
    L = 25              # Lattice size
    beta = 0.1          # Inverse temperature
    h = 0.05           # Step size
    n = 5              # Number of velocity verlet steps.
    Nsteps = int(1E4)   # Number of MCMC steps
    metropolize = False # Do metropolize or not.

    if metropolize:
        [xy, rej_rate, mags] = HybridMC(L, beta, h, n, Nsteps, True, True)
        print("\nMetropolized Scheme")
        print("\nRejection Rate = {:.2f}%".format(100*rej_rate))

    else:
        print("\nUn-Metropolized Scheme")
        [xy, mags] = HybridMC(L, beta, h, n, Nsteps, False, True)

    acf = acor.function(mags)

    # Time for the correlation to first reach 0 (within the tolerance).
    cor_time = np.where(acf <= 1E-6)[0][0]

    plt.figure(3)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    if metropolize:
        plt.title('ACF of the Metropolized Scheme')
    else:
        plt.title('ACF of the Un-Metropolized Scheme')
    plt.plot(np.arange(cor_time+1), acf[:cor_time+1], 'b-')

    tau = acor.acor(mags, maxlag = cor_time)[0]
    print("\nIAC = {:.1f}\n".format(tau))

test_IAC()
plt.show()
