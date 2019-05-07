"""
File: ex43.py

Author: Terrence Alsup
Date: March 27, 2019
Monte Carlo Methods HW 3

Run script to produce plots from exercise 43.
"""

import numpy as np
import acor
import IsingSamplers
from matplotlib import pyplot as plt


L = 10
beta = 0.05
numParticles = 25
N = 10

##
## Evaluate performace of the estimator.
##
numTrials = 100
trials = np.zeros(numTrials)
trialsSS = np.zeros(numTrials)
for k in range(numTrials):
    [X,W] = IsingSamplers.Jarzynski(N, beta, L, numParticles, resampling=False)
    sum = 0.
    for j in range(numParticles):
        sum += IsingSamplers.magnetization(X[j])*W[j]
    trials[k] = sum
    trialsSS[k] = 1/np.sum(W**2)

print("\n")
print("N = {0}, Number of Particles = {1}, Number of Trials = {2}".format(N, numParticles, numTrials))
print("Estimated Mean Magnetization:          ", np.mean(trials))
print("Variance of Estimated Magnetization:   ", np.std(trials)**2)
print("Estimated Effective Sample Size Ratio: ", np.mean(trialsSS)/numParticles)
print("\n")

##
## Examine how chaning N affects the variance.
##

Ns = [5, 10, 15, 20, 25, 30, 35]
variances = np.zeros(len(Ns))
variancesR = np.zeros(len(Ns))

L = 10
beta = 1.
numParticles = 25

numTrials = 100

for i in range(len(Ns)):
    N = Ns[i]
    trials = np.zeros(numTrials)
    trialsSS = np.zeros(numTrials)
    for k in range(numTrials):
        [X,W] = IsingSamplers.Jarzynski(N, beta, L, numParticles, resampling=False)
        sum = 0.
        for j in range(numParticles):
            sum += IsingSamplers.magnetization(X[j])*W[j]
        trials[k] = sum
        trialsSS[k] = 1/np.sum(W**2)
    variances[i] = np.std(trials)**2

for i in range(len(Ns)):
    N = Ns[i]
    trials = np.zeros(numTrials)
    trialsSS = np.zeros(numTrials)
    for k in range(numTrials):
        [X,W] = IsingSamplers.Jarzynski(N, beta, L, numParticles, resampling=True)
        sum = 0.
        for j in range(numParticles):
            sum += IsingSamplers.magnetization(X[j])*W[j]
        trials[k] = sum
        trialsSS[k] = 1/np.sum(W**2)
    variancesR[i] = np.std(trials)**2

plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(Ns, variances, 'b-x',label='Without Resampling')
plt.plot(Ns, variancesR, 'r-s', label='With Resampling')
plt.xlabel('$N$')
plt.ylabel('Variance')
plt.title('Variance of the Magnetization Estimator vs. $N$')
plt.legend()
plt.show()
