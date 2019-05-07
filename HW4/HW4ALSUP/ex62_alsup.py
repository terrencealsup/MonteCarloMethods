"""
File: ex62_alsup.py
Author: Terrence Alsup
Date: April 17, 2019
Monte Carlo Methods HW 4

Sample from the XY model using a metropolized and un-metropolized underdamped
Langevin sampler.  Plot 2 samples at different temperatures.  Compute the IACs
for the cosine of the angle of the magnetization vector.
Does Exercise 62 in the notes.
"""
import numpy as np
from numpy import linalg
import XYmodel
import acor
from matplotlib import pyplot as plt

def log_r(xy, Xh, Xt, Yh, Yt, h, gamma):
    """
    Evaluate log r(x,y)
    """
    v = Yt - np.exp(-gamma*h)*Xt
    v -= 0.5*h*(np.exp(-gamma*h)*xy.grad_log_density(Xh) + xy.grad_log_density(Yh))
    logr = -0.5*linalg.norm(v)**2 / (1 - np.exp(-2*h*gamma))
    return logr

def log_pacc(xy, Xh, Xt, Yh, Yt, h, gamma):
    """
    Evaluate the logarithm of the acceptance probability for Metropolis.
    """
    log_piy = xy.log_density(Yh) - 0.5*linalg.norm(Yt)**2
    log_pix = xy.log_density(Xh) - 0.5*linalg.norm(Xt)**2
    log_ryx = log_r(xy, Yh, -Yt, Xh, -Xt, h, gamma)
    log_rxy = log_r(xy, Xh, Xt, Yh, Yt, h, gamma)

    return log_piy + log_ryx - log_pix - log_rxy


def underdampedLangevinStep(xy, Xh, Xt, gamma, h):
    """
    Compute a single step of the underdamped Langevin dynamics.
    """
    Xt_temp = Xt + (0.5 * h * xy.grad_log_density(Xh))
    Xh_temp = Xh + (0.5 * h * Xt_temp)
    Xt_temptemp = np.exp(-gamma*h) * Xt_temp + np.sqrt(1 - np.exp(-2*gamma*h))*np.random.randn(xy.L)
    Xhp1 = Xh_temp + 0.5*h*Xt_temptemp
    Xtp1 = Xt_temptemp + 0.5*h*xy.grad_log_density(Xhp1)
    return [Xhp1, Xtp1]


def underdampedLangevin(L, beta, h, gamma, Nsteps, metropolize=False, getMags=False):
    """
    Sample the XY model using an underdamped Langevin sampler.
    """
    xy = XYmodel.XYmodel(L, beta)

    if getMags:
        mag = np.zeros(Nsteps+1)
        mag[0] = xy.cosMagnetVector()

    if metropolize:
        rejected = 0

    # Get the initial position variables.
    Xh = xy.theta

    # Randomly sample the initial momentum variables from exp(-K(x)).
    Xt = np.random.randn(xy.L)


    # Do Nsteps of MCMC.
    for k in range(1, Nsteps + 1):

        # Run the underdamped Langevin dynamics.
        [Yh, Yt] = underdampedLangevinStep(xy, Xh, Xt, gamma, h)

        if metropolize:
            # Use log probability to avoid under/overflow
            if np.log(np.random.rand()) < log_pacc(xy, Xh, Xt, Yh, Yt, h, gamma):
                # Note that we do not need to reset the momentum variables
                # because we sample them at each step independently.
                Xh = Yh
                Xt = Yt
                xy.set(Xh)
            else:
                Xt = -Xt
                rejected += 1
        else:
            Xh = Yh
            Xt = Yt
            xy.set(Xh)

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
    beta = 1.0         # Inverse temperature
    h = 1E-1            # Step size
    gamma = 0.1         # Friction coefficient
    Nsteps = int(1E4)   # Number of MCMC steps

    # First plot the spins on the circle for large beta.

    # Get points on the circle.
    angle = np.linspace(0, 2*np.pi, 1000)
    circle = [np.cos(angle), np.sin(angle)]

    # Sample from the model and get the spin vectors.
    [xy, rej_rate] = underdampedLangevin(L, beta, h, gamma, Nsteps, metropolize=True)

    print("\nRejection Rate = {:.2f}%\n".format(rej_rate*100))

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
    beta = 0.1         # Inverse temperature
    h = 1E-1            # Step size
    gamma = 5E2         # Friction coefficient
    Nsteps = int(1E4)   # Number of MCMC steps
    metropolize = False # Do metropolize or not.

    if metropolize:
        [xy, rej_rate, mags] = underdampedLangevin(L, beta, h, gamma, Nsteps, True, True)
        print("\nMetropolized Scheme")
        print("\nRejection Rate = {:.2f}%".format(100*rej_rate))

    else:
        print("\nUn-Metropolized Scheme")
        [xy, mags] = underdampedLangevin(L, beta, h, gamma, Nsteps, False, True)


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
