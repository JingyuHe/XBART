import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xbart import XBART
import time
seed = 98765
np.random.seed(98765)


def linear(x):
    d = x.shape[1]
    beta = [-2 + 4*(i - 1) / (d-1) for i in range(1, d+1)]
    return x.dot(beta)
    
def single_index(x):
    d = x.shape[1]
    gamma = [-1.5 + i/3 for i in list(range(0, d))]
    a =  np.apply_along_axis(lambda x: sum((x-gamma)**2), 1, x)
    f = 10 * np.sqrt(a) + np.sin(5*a)
    return f

def trig_poly(x):
    f = np.apply_along_axis(lambda x: 5 * np.sin(3*x[0]) + 2 * x[1]**2 + 3 * x[2] * x[3], 1, x)
    return f

def xbart_max(x):
    return np.apply_along_axis(lambda x: max(x[0:2]), 1, x)

def generate_data(x, dgp):
    if dgp == 'linear':
        return linear(x)
    if dgp == 'single_index':
        return single_index(x)
    if dgp == 'trig_poly':
        return trig_poly(x)
    if dgp == 'max':
        return xbart_max(x)


if __name__ == "__main__":
    # simulation
    n = 1000
    d = 5
    dgp = 'single_index'
    nt = 50
    nrep = 10
    alpha = 0.1

    X = np.random.normal(size=(n,d))
    Y = generate_data(X, dgp) + np.random.normal(size=n)

    min_Y = Y.min() - 0.1 * (Y.max()-Y.min())
    max_Y = Y.max() + 0.1 * (Y.max()-Y.min())

    # sigma = 0.1*sqrt(kappa^2 * var(f))
    # y = f + rnorm(n, 0, sigma)

    # x_range <- sapply(1:d, function(i, x) max(x[,i]) - min(x[,i]), x)

    # replicate test set
    X1 = np.zeros((nt * nrep, d))
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), nt)
    X1[:,0] = np.repeat(x1, nrep)
    Y1 = generate_data(X1, dgp) + np.random.normal(size=nt * nrep)

    num_trees = 10
    num_sweeps = 200
    tau = np.var(Y) / num_trees
    theta = 0.1
    xbart = XBART(num_trees = num_trees, num_sweeps = num_sweeps, burnin = 15, tau = tau, sampling_tau = True)
    xbart.fit(X,Y,0)
    mu_pred = xbart.predict_gp(X, Y, X1, p_cat = 0, theta = theta, tau = tau, return_mean=False)

    y_pred = pd.DataFrame(mu_pred).transpose().apply(
        lambda x: x + xbart.sigma_draws[:,num_trees - 1] * np.random.normal(size=num_sweeps), 0).transpose() 
    xbart_PI_gp =  pd.DataFrame(y_pred).transpose().apply(
                    lambda x: np.quantile(x, [alpha/2, 1 - alpha/2]), 0).transpose()
    xbart_PI_gp.rename(columns = {0: 'lower', 1: 'upper'}, inplace = True)

    mu_pred = xbart.predict(X1, return_mean = False)
    y_pred = pd.DataFrame(mu_pred).transpose().apply(
        lambda x: x + xbart.sigma_draws[:,num_trees - 1] * np.random.normal(size=num_sweeps), 0).transpose() 
    xbart_PI =  pd.DataFrame(y_pred).transpose().apply(
                    lambda x: np.quantile(x, [alpha/2, 1 - alpha/2]), 0).transpose()
    xbart_PI.rename(columns = {0: 'lower', 1: 'upper'}, inplace = True)

    print('xbart width = ' + str((xbart_PI['upper'] - xbart_PI['lower']).mean()))
    print('gp width = ' + str((xbart_PI_gp['upper'] - xbart_PI_gp['lower']).mean()))

