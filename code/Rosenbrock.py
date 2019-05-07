import numpy as np
from matplotlib import pyplot as plt

def pi(x1, x2):
    """
    Compute the Rosenbrock density up to a normalizing constant.
    """
    return np.exp( -5*(x2 - x1**2)**2 - 0.05*(1 - x1)**2)

def log_pi(x1, x2):
    """
    Computes the log of the Rosenbrock density.
    """
    return ( -5*(x2 - x1**2)**2 - 0.05*(1 - x1)**2 )

def grad_log_pi(x1, x2):
    """
    Computes the gradient of the log density.
    """
    g1 = 20 * x1 * (x2 - x1**2) + 0.01 * (1 - x1)
    g2 = -10 * (x2 - x1**2)
    return np.asarray([g1, g2])

def D2_log_pi(x1, x2):
    """
    Computes minus the Hessian of the log density.
    """
    H11 = 20 * (x2 - x1**2) - 40 * x1**2 - 0.01
    H12 = 20 * x1 # Note that H_{12} = H_{21}
    H22 = -10
    return -np.asarray([[H11, H12], [H12, H22]])


def plot_density():
    """
    Plot the Rosenbrock density.
    """
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-1, 5, 100)

    X1, X2 = np.meshgrid(x1, x2, sparse=True)

    Y = pi(X1, X2)

    plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.contourf(x1, x2, Y)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Rosenbrock Density')
    plt.show()
