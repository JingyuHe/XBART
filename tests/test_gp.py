import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xbart import XBART
import time
seed = 98765
np.random.seed(98765)

if __name__ == "__main__":
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
    xbart = XBART(num_trees = num_trees, num_sweeps = num_sweeps, burnin = 15, tau = tau, sampling_tau = True, 
        set_random_seed = True, seed = 98765)
    xbart.fit(X,Y,0)
    mu_pred = xbart.predict_gp(X, Y, X1, p_cat = 0, theta = theta, tau = tau, return_mean=False)

    y_pred = pd.DataFrame(mu_pred).transpose().apply(
        lambda x: x + xbart.sigma_draws[:,num_trees - 1] * np.random.normal(size=num_sweeps), 0).transpose() 

    bound =  pd.DataFrame(y_pred).transpose().apply(
                        lambda x: np.quantile(x, [alpha/2, 1 - alpha/2]), 0).transpose()
    bound.rename(columns = {0: 'lower', 1: 'upper'}, inplace = True)

    coverage = ((bound['lower'] <= Y1)&(bound['upper'] >= Y1)).mean()
    width = (bound['upper'] - bound['lower']).mean()
    print('coverage = ' + str(coverage))
    print('interval width = ' + str(round(width, 3)))


