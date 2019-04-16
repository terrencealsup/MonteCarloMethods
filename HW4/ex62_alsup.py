import numpy as np
import XYmodel
import acor

def step():
    Xh


def underdampedLangevin(L, beta, h, Nsteps):
    """
    Sample the XY model using an underdamped Langevin sampler.
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
            temp = -xy.grad_log_density(yh) + 0.5 * linalg.norm(yt)
            temp -= (-xy.grad_log_density() + 0.5 * linalg.norm(yt_old))
            p_acc = np.exp(temp)
            if np.random.rand() < p_acc:
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
