import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xbart import XBART
import time
seed = 98765
np.random.seed(98765)


run_full_CP = False
# flag for whether to run full conformal prediction
# requires the nonconformist library

TOL = 1e-8

def leastsq_minL2(X,Y,X1,tol=TOL):
    uX,dX,vX = np.linalg.svd(X)
    rX = (dX>=dX[0]*tol).sum()
    betahat = (vX[:rX].T/dX[:rX]).dot(uX[:,:rX].T.dot(Y))
    return X1.dot(betahat)

# simulation
n = 100; 
n1 = 100
SNR = 10
ntrial = 50
alpha = 0.1
dim_vals = np.arange(5,205,5)
d = dim_vals[0]

beta = np.random.normal(size=d)
beta = beta/np.sqrt((beta**2).sum()) * np.sqrt(SNR)
X = np.random.normal(size=(n,d))
Y = X.dot(beta) + np.random.normal(size=n)

min_Y = Y.min() - 0.1 * (Y.max()-Y.min())
max_Y = Y.max() + 0.1 * (Y.max()-Y.min())

X1 = np.random.normal(size=(n1,d))
Y1 = X1.dot(beta) + np.random.normal(size=n1)

num_trees = 10
num_sweeps = 1000
tau = np.var(Y) / num_trees
theta = 10
xbart = XBART(num_trees = num_trees, num_sweeps = 200, burnin = 15, tau = tau, sampling_tau = True, 
             set_random_seed = True, seed = 98765)
time_start_fit = time.time()
xbart.fit(X,Y,0)
time_start_predict = time.time()
y_pred = xbart.predict_gp(X, Y, X1, p_cat = 0, theta = theta, tau = tau, return_mean=False)
time_end_predict = time.time()
y_hat_xbart = y_pred[:,15:].mean(axis=1)


