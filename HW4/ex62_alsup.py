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
    beta = 10.0         # Inverse temperature
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


test_sampler()
plt.show()
