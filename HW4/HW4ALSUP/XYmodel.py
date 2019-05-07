"""
Author: Terrence Alsup
Date: April 13, 2019
Monte Carlo Methods HW 4
File: XYmodel.py

Implement the XY model as a Python class and include methods to compute the
magnetization vector as well as the cosine of the angle of the magnetizaion
vector.  Also contains methods to evaluate the log density and its gradient.
"""
import numpy as np

class XYmodel:
    """
    An instance of the XY model.
    """
    def __init__(self, L, beta):
        """
        Initialize the XY model with uniform random angles.
        """
        self.L = L          # Lattice size
        self.beta = beta    # Inverse temperature
        self.theta = 2*np.pi*np.random.rand(L)  # Uniform random angles.

    def spinVectors(self):
        """
        Return an L-by-2 vector of the spins at each lattice site.
        """
        spinVec = np.zeros((self.L, 2))
        spinVec[:,0] = np.cos(self.theta)
        spinVec[:,1] = np.sin(self.theta)
        return spinVec


    def magnetization(self):
        """
        Compute the magnetization vector in R^2.
        """
        sigma_x = np.cos(self.theta)
        sigma_y = np.sin(self.theta)
        return np.asarray([np.sum(sigma_x), np.sum(sigma_y)])

    def cosMagnetVector(self):
        """
        Return the cosine of the angle of the magnetization vector.
        """
        mag = self.magnetization()
        return mag[0]/np.sqrt(mag[0]**2 + mag[1]**2)

    def log_density(self, theta=None):
        """
        Compute the log of the density up to its normalizing constant.
        beta sum_{i <-> j} cos(theta_i - theta_j)
        """

        if theta is None:
            theta = self.theta

        sum = 0.
        for i in range(self.L):
            sum += np.cos(theta[np.mod(i+1, self.L)] - theta[i])

        return (self.beta * sum)

    def grad_log_density(self, theta=None):
        """
        Return the gradient of log pi w.r.t. theta as a vector of size L.
        """
        grad = np.zeros(self.L)
        for i in range(self.L):
            if theta is not None:
                grad[i] = np.sin(theta[np.mod(i-1, self.L)] - theta[i])
                grad[i] += np.sin(theta[np.mod(i+1, self.L)] - theta[i])
            else:
                grad[i] = np.sin(self.theta[np.mod(i-1, self.L)] - self.theta[i])
                grad[i] += np.sin(self.theta[np.mod(i+1, self.L)] - self.theta[i])

        return (self.beta * grad)

    def density_unnormalized(self):
        return np.exp(self.log_density())

    def set(self, new_theta):
        """
        Set new_theta as the current vector of thetas.
        """
        self.theta = new_theta
