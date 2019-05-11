import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import Rosenbrock

def UnderdampedSampler(x0, h, N, J_type='id'):
    # Functions to evaluate the density, grad log density, and Hessian of the
    # minus log density.
    pi = Rosenbrock.pi
    log_pi = Rosenbrock.log_pi
    grad_log_pi = Rosenbrock.grad_log_pi
    D2_log_pi = Rosenbrock.D2_log_pi
    J = Rosenbrock.S
    divJ = Rosenbrock.divS

    MCMC_chain = np.zeros((N+1, len(x0)))
    MCMC_chain[0,:] = x0

    x = x0
    accept = 0
    for k in range(1, N+1):
        y = update(x, h, pi, grad_log_pi, S, divS, S_type)
        [x, is_accepted] = metropolize(x, y, h, pi, log_pi, grad_log_pi, S, divS, S_type)
        if is_accepted:
            accept += 1
        MCMC_chain[k,:] = x

    return [MCMC_chain, accept/N]

def update(x, h, pi, grad_log_pi, S, divS, S_type):
    if S_type == 'id':
        return ( x + h*grad_log_pi(x) + np.sqrt(2*h)*np.random.randn(len(x)) )
    else:
        # S = ( - D^2 log pi)^{-1}
        [S1, sigma] = S(x)
        y = x + h * (S1 @ grad_log_pi(x))
        y += h * divS(x)
        y += np.sqrt(2 * h) * (sigma @ np.random.randn(len(x)))
        return y

def metropolize(x, y, h, pi, log_pi, grad_log_pi, S, divS, S_type):
    log_pix = log_pi(x)
    log_piy = log_pi(y)

    if S_type == 'id':
        mux = x + h*grad_log_pi(x)
        muy = y + h*grad_log_pi(y)
        log_qyx = -linalg.norm(y - mux)**2 / (4 * h)
        log_qxy = -linalg.norm(x - muy)**2 / (4 * h)
    else:
        # S = (-D^2 log pi)^{-1}
        [S1, sigma1] = S(x)
        divS1 = divS(x)
        [S2, sigma2] = S(y)
        divS2 = divS(y)
        mux = x + h*grad_log_pi(x) + h*divS1
        muy = y + h*grad_log_pi(y) + h*divS2
        log_qyx = -(x - mux).transpose() @ linalg.inv(S1) @ (x-mux)/ (4*h)
        log_qxy = -(y - muy).transpose() @ linalg.inv(S2) @ (y-muy)/ (4*h)

    log_pacc = log_qxy + log_piy - log_qyx - log_pix
    U = np.random.rand()
    if np.log(U) < log_pacc:
        # Accepted.
        return [y, True]
    else:
        # Rejected.
        return [x, False]

def plot_trajectory(x0, h, N, S_type='id'):
    [chain, accept] = UnderdampedSampler(x0, h, N, S_type)

    print("\nAcceptance Rate = {:0.1f}%.\n".format(100*accept))

    pi = Rosenbrock.pi
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-2, 6, 100)

    X1, X2 = np.meshgrid(x1, x2, sparse=True)

    Y = pi([X1, X2])

    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.contourf(x1, x2, Y)
    plt.plot(chain[:,0], chain[:,1], 'r')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Underdamped Langevin Trajectory')
    plt.show()

plot_trajectory([0,2], 0.1, 1000, S_type = 'Hessian')
