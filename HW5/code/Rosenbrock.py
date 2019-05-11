import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt

def pi(x):
    """
    Compute the Rosenbrock density up to a normalizing constant.
    """
    x1 = x[0]
    x2 = x[1]
    return np.exp( -5*(x2 - x1**2)**2 - 0.05*(1 - x1)**2)

def log_pi(x):
    """
    Computes the log of the Rosenbrock density.
    """
    x1 = x[0]
    x2 = x[1]
    return ( -5*(x2 - x1**2)**2 - 0.05*(1 - x1)**2 )

def grad_log_pi(x):
    """
    Computes the gradient of the log density.
    """
    x1 = x[0]
    x2 = x[1]
    g1 = 20 * x1 * (x2 - x1**2) + 0.01 * (1 - x1)
    g2 = -10 * (x2 - x1**2)
    return np.asarray([g1, g2])

def D2_log_pi(x):
    """
    Computes minus the Hessian of the log density.
    """
    x1 = x[0]
    x2 = x[1]
    H11 = 20 * (x2 - x1**2) - 40 * x1**2 - 0.01
    H12 = 20 * x1 # Note that H_{12} = H_{21}
    H22 = -10
    return -np.asarray([[H11, H12], [H12, H22]])

def S(x):
    """
    Computes sqrt(S) = sigma, where S = (-D^2 log pi)^{-1}
    Positive square-root.
    Also returns the absolute value of S = P|D|P^T
    """
    Sinv = D2_log_pi(x)
    evals, P = linalg.eig(Sinv)
    sqrtD = np.diag(np.sqrt(np.abs(1/evals)))
    sigma = P @ sqrtD
    absS  = sigma @ sigma.transpose()
    return [absS, P @ sqrtD]

def divS(x):
    """
    Numerically compute the divergence of S(x).
    """
    d = len(x) # Get the dimension of x.
    divS = np.zeros(d)
    dx = 1e-6 # Step used for finite differences.
    for j in range(d):
        e = np.zeros(d)
        e[j] = 1
        Spdx = S(x + dx*e)[0]
        Smdx = S(x - dx*e)[0]

        # Second-order centered difference for first derivative D_j S.
        divS += (Spdx[:,j] - Smdx[:,j]) / (2 * dx)

    return divS






def plot_density():
    """
    Plot the Rosenbrock density.
    """
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-1, 5, 100)

    X1, X2 = np.meshgrid(x1, x2, sparse=True)

    Y = pi([X1, X2])

    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.contourf(x1, x2, Y)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Rosenbrock Density')
    plt.show()
