"""
File: ex60_alsup.py
Author: Terrence Alsup
Date: April 17, 2019
Monte Carlo Methods HW 4


"""
import XYmodel
import numpy as np
import acor
from matplotlib import pyplot as plt
from numpy import linalg



def sampleXY(L, beta, h, Nsteps, metropolize=False, getMags=False):
    """
    Generate a sample from the XY model.

    S = I
    xi ~ N(0, I)
    """
    xy = XYmodel.XYmodel(L, beta)

    if getMags:
        mag = np.zeros(Nsteps+1)
        mag[0] = xy.cosMagnetVector()

    if metropolize:
        rejected = 0

    for k in range(1,Nsteps+1):

        hgradLogPi = h * xy.grad_log_density() # Drift term.
        dX = hgradLogPi + (np.sqrt(h) * np.random.randn(L))
        Y = dX + xy.theta

        if metropolize:
            qxy = np.exp(linalg.norm(-dX - hgradLogPi)**2/(4*h))    # q(x|y)
            qyx = np.exp(linalg.norm( dX - hgradLogPi)**2/(4*h))    # q(y|x)
            X = xy.theta # The current value of theta.
            pix = xy.density_unnormalized()
            xy.set(Y) # Go ahead and set the values of theta.
            piy = xy.density_unnormalized()
            p_acc = (qxy * piy)/(qyx * pix) # Acceptance probability.
            # If rejected, reset to the old value.
            if np.random.rand() > p_acc:
                xy.set(X)
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
    h = 1E-3            # Step size
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
    Test the un-metropolized XY model sampler and get the ACF and IAC.

    Nsteps is the number of MCMC steps.
    """
    L = 25              # Lattice size
    beta = 1.0          # Inverse temperature
    h = 1E-1            # Step size
    Nsteps = int(1E4)   # Number of MCMC steps

    [xy, mags] = sampleXY(L, beta, h, Nsteps, getMags=True)

    acf = acor.function(mags)

    # Time for the correlation to first reach 0 (within the tolerance).
    cor_time = np.where(acf <= 1E-6)[0][0]

    plt.figure(3)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('ACF of the Un-Metropolized Scheme')
    plt.plot(np.arange(cor_time+1), acf[:cor_time+1], 'b-')

    tau = acor.acor(mags, maxlag = cor_time)[0]
    print("\nIAC = {:.1f}\n".format(tau))


test_IAC()
plt.show()
