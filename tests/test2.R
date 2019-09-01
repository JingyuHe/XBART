#######################################################################
# set parameters of XBART
get_XBART_params <- function(n, d, y) {
  XBART_params = list(num_trees = 10,                 # number of trees 
                      num_sweeps = 40,           # number of sweeps (samples of the forest)
                      n_min = 1,               # minimal node size
                      alpha = 0.95,           # BART prior parameter 
                      beta = 1.25,            # BART prior parameter
                      mtry = 1,               # number of variables sampled in each split
                      burnin = 15,
                      no_split_penality = "Auto"
                      )            # burnin of MCMC sample
  num_tress = XBART_params$num_trees
  XBART_params$max_depth = matrix(250, num_tress, XBART_params$num_sweeps)   # max depth of each tree, should be a num_trees by num_sweeps matrix
  XBART_params$num_cutpoints = 50;                                           # number of adaptive cutpoints
  XBART_params$tau = var(y) / num_tress                                   # prior variance of mu (leaf parameter)
  return(XBART_params)
}

#######################################################################
library(XBART)

set.seed(100)
d = 1 # number of TOTAL variables
dcat = 0 # number of categorical variables




n = 10000 # size of training set
nt = 5000 # size of testing set

new_data = TRUE # generate new data
run_dbarts = FALSE # run dbarts
run_xgboost = FALSE # run xgboost
run_lightgbm = FALSE # run lightgbm
parl = TRUE # parallel computing



x = rnorm(n)
xtest = rnorm(nt)

y = sin(3 * x)
ytest = sin(3 * xtest)

sigma = sd(y)

y = y + rnorm(n) * sigma
ytest = ytest + rnorm(nt) * sigma



params = get_XBART_params(n, d, y)
time = proc.time()
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), p_categorical = dcat,
            params$num_trees, params$num_sweeps, params$max_depth,
            params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
            mtry = params$mtry, draw_mu = TRUE,
            num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100,no_split_penality=params$no_split_penality)
time = proc.time() - time

fhat.1 = apply(fit$yhats_test[, params$burnin:params$num_sweeps], 1, mean)
fhat = apply(fit$yhats[, params$burnin:params$num_sweeps], 1, mean)


plot(fhat, y)

rmse = sqrt(mean((ytest - fhat.1)^2))


print(rmse)
print(time)
