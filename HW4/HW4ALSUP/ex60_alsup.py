"""
File: ex60_alsup.py
Author: Terrence Alsup
Date: April 17, 2019
Monte Carlo Methods HW 4

Sample from the XY model using a metropolized and un-metropolized stochastic
thermostat.  Plot 2 samples at different temperatures.  Compute the IACs for
the cosine of the angle of the magnetization vector.  Change the size of L.
Does Exercise 60 in the notes.
"""
import XYmodel
import numpy as np
import acor
from matplotlib import pyplot as plt
from numpy import linalg


def log_q(xy, x, y, h):
    """
    Compute the logarithm of the proposal log q(y|x).
    """
    return -linalg.norm(y - x - h*xy.grad_log_density(x))/(4*h)

def log_pacc(xy, x, y, h):
    """
    Compute the logarithm of the acceptance probability.
    """
    logp = log_q(xy, y, x, h)
    logp += xy.log_density(y)
    logp -= log_q(xy, x, y, h)
    logp -= xy.log_density(x)
    return logp

def sampleXY(L, beta, h, Nsteps, metropolize=False, getMags=False):
    """
    Generate a sample from the XY model.
    """
    xy = XYmodel.XYmodel(L, beta)

    if getMags:
        mag = np.zeros(Nsteps+1)
        mag[0] = xy.cosMagnetVector()

    if metropolize:
        rejected = 0

    for k in range(1,Nsteps+1):

        dX =  h * xy.grad_log_density() + (np.sqrt(h) * np.random.randn(L))
        Y = dX + xy.theta

        if metropolize:
            # Compute acceptance probability.
            if np.log(np.random.rand()) < log_pacc(xy, xy.theta, Y, h):
                xy.set(Y)
            else:
                rejected += 1
        else:
            xy.set(Y)
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
    beta = 10.0         # Inverse temperature
    h = 1E-2            # Step size
    Nsteps = int(1E4)   # Number of MCMC steps

    # First plot the spins on the circle for large beta.

    # Get points on the circle.
    angle = np.linspace(0, 2*np.pi, 1000)
    circle = [np.cos(angle), np.sin(angle)]

    # Sample from the model and get the spin vectors.
    xy = sampleXY(L, beta, h, Nsteps)
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

    # Now plot the spins for small beta.
    beta = 0.1 # Inverse temperature

    # Sample from the model and get the spin vectors.
    xy = sampleXY(L, beta, h, Nsteps)
    spins = xy.spinVectors()

    # Now plot the result.
    plt.figure(2)
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
    Test the XY model sampler and get the ACF and IAC.

    Nsteps is the number of MCMC steps.
    """
    L = 25              # Lattice size
    beta = 0.1          # Inverse temperature
    h = 0.5             # Step size
    Nsteps = int(1E4)   # Number of MCMC steps
    metropolize = True  # Do metropolize or not.

    if metropolize:
        [xy, rej_rate, mags] = sampleXY(L, beta, h, Nsteps, True, True)
        print("\nMetropolized Scheme")
        print("\nRejection Rate = {:.2f}%".format(100*rej_rate))

    else:
        print("\nUn-Metropolized Scheme")
        [xy, mags] = sampleXY(L, beta, h, Nsteps, False, True)

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

test_sampler()
test_IAC()
plt.show()
