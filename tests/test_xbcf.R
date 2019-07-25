#######################################################################
# set parameters of XBART
get_XBCF_params <- function(n, d, y) {
  XBCF_params = list(num_sweeps = 40,           # number of sweeps (samples of the forest)
                      burnin = 15,
                      n_min = 1,               # minimal node size
                      num_trees_pr = 20,
                      alpha_pr = 0.95,           # BART prior parameter 
                      beta_pr = 2,            # BART prior parameter
                      num_trees_trt = 5,
                      alpha_trt = 0.25,           # BART prior parameter 
                      beta_trt = 3,            # BART prior parameter
                      mtry = 10,               # number of variables sampled in each split
                      no_split_penality = "Auto"
  )            # burnin of MCMC sample
  num_tress_pr = XBCF_params$num_trees_pr
  num_tress_trt = XBCF_params$num_trees_trt
  XBCF_params$max_depth = 250
  XBCF_params$num_cutpoints = 50;                                           # number of adaptive cutpoints
  XBCF_params$tau_pr = var(y) / num_tress_pr
  XBCF_params$tau_trt = var(y) / num_tress_trt
  XBCF_params$p_categorical = 10
  # prior variance of mu (leaf parameter)
  return(XBCF_params)
}


#######################################################################
library(XBART)

set.seed(100)
d = 20 # number of TOTAL variables
dcat = 10 # number of categorical variables
# must be d >= dcat

# (X_continuous, X_categorical), 10 and 10 for each case, 20 in total



n = 1000 # size of training set
nt = 500 # size of testing set

new_data = TRUE # generate new data
run_dbarts = FALSE # run dbarts
run_xgboost = FALSE # run xgboost
run_lightgbm = FALSE # run lightgbm
parl = TRUE # parallel computing


#######################################################################
# Data generating process

#######################################################################
# Have to put continuous variables first, then categorical variables  #
# X = (X_continuous, X_cateogrical)                                   #
#######################################################################
if (new_data) {
  if (d != dcat) {
    x = matrix(runif((d - dcat) * n, -2, 2), n, d - dcat)
    if (dcat > 0) {
      x = cbind(x, matrix(as.numeric(sample(-2:2, dcat * n, replace = TRUE)), n, dcat))
    }
  } else {
    x = matrix(as.numeric(sample(-2:2, dcat * n, replace = TRUE)), n, dcat)
  }
  
  
  if (d != dcat) {
    xtest = matrix(runif((d - dcat) * nt, -2, 2), nt, d - dcat)
    if (dcat > 0) {
      xtest = cbind(xtest, matrix(as.numeric(sample(-2:2, dcat * nt, replace = TRUE)), nt, dcat))
    }
  } else {
    xtest = matrix(as.numeric(sample(-2:2, dcat * nt, replace = TRUE)), nt, dcat)
  }
  
  f = function(x) {
    sin(rowSums(x[, 3:4] ^ 2)) + sin(rowSums(x[, 1:2] ^ 2)) + (x[, 15] + x[, 14]) ^ 2 * (x[, 1] + x[, 2] ^ 2) / (3 + x[, 3] + x[, 14] ^ 2)
    #rowSums(x[,1:30]^2)
    #pmax(x[,1]*x[,2], abs(x[,3])*(x[,10]>x[,15])+abs(x[,4])*(x[,10]<=x[,15]))
    #
  }
  
  # to test if ties cause a crash in continuous variables
  x[, 1] = round(x[, 1], 4)
  #xtest[,1] = round(xtest[,1],2)
  ftrue = f(x)
  ftest = f(xtest)
  sigma = sd(ftrue)
  
  #y = ftrue + sigma*(rgamma(n,1,1)-1)/(3+x[,d])
  #y_test = ftest + sigma*(rgamma(nt,1,1)-1)/(3+xtest[,d])
  
  y = ftrue + sigma * rnorm(n)
  y_test = ftest + sigma * rnorm(nt)
}

#######################################################################
# XBART
categ <- function(z, j) {
  q = as.numeric(quantile(x[, j], seq(0, 1, length.out = 100)))
  output = findInterval(z, c(q, + Inf))
  return(output)
}


params = get_XBCF_params(n, d, y)
p_categorical = dcat
z <- rbinom(length(y), 1, 0.5)
time = proc.time()
fit = XBCF(as.matrix(y), as.matrix(x), as.matrix(z), 
           params$num_sweeps, params$burnin,
           params$max_depth, params$n_min,
           params$num_cutpoints,
           params$no_split_penality, params$mtry,
           params$p_categorical,
           params$num_trees_pr, params$alpha_pr, params$beta_pr, params$tau_pr, 1, 1, FALSE,
           params$num_trees_trt, params$alpha_trt, params$beta_trt, params$tau_trt, 1, 1, FALSE,
           random_seed = 100)


print(paste("XBCF output: ", fit))
