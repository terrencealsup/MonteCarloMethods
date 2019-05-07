import numpy as np
from matplotlib import pyplot as plt
import Rosenbrock

class OverdampedSampler:
    """
    Create a sampler to sample from the Rosenbrock density.
    """
    def __init__(self, h, N, S_type = 'id'):
        self.pi = Rosenbrock.pi
        self.grad_log_pi = Rosenbrock.grad_log_pi


    def update(self, x):
        h = self.h
        S =
        y = x
        y +=

    def divS(self, x):
        """
        Compute the divergence of the matrix S at the point x.
        """
        if self.S_type == 'id':
            return np.zeros(2)
        else:
            
