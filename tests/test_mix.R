#######################################################################
# set parameters of XBART
get_XBART_params <- function(y) {
  XBART_params = list(num_trees = 30, # number of trees 
                      num_sweeps = 40, # number of sweeps (samples of the forest)
                      n_min = 1, # minimal node size
                      alpha = 0.95, # BART prior parameter 
                      beta = 1.25, # BART prior parameter
                      mtry = 10, # number of variables sampled in each split
                      burnin = 15,
                      no_split_penality = "Auto"
                      ) # burnin of MCMC sample
  num_tress = XBART_params$num_trees
  XBART_params$max_depth = 250
  XBART_params$num_cutpoints = 50;
  # number of adaptive cutpoints
  XBART_params$tau = var(y) / num_tress # prior variance of mu (leaf parameter)
  return(XBART_params)
}

#######################################################################
library(XBART)

set.seed(100)
new_data = TRUE # generate new data
parl = FALSE # parallel computing

small_case = TRUE # run simulation on small data set
verbose = FALSE # print the progress on screen

if (small_case) {
  n = 20000 # size of training set
  nt = 5000 # size of testing set
  d = 20 # number of TOTAL variables
  dcat = 10 # number of categorical variables
  p_z = 5
  # must be d >= dcat
  # (X_continuous, X_categorical), 10 and 10 for each case, 20 in total
} else {
  n = 100000
  nt = 10000
  d = 50
  dcat = 0
  p_z = 5
}


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

  z = matrix(rnorm(p_z * n), n, p_z)
  ztest = matrix(rnorm(p_z * nt), nt, p_z)

  theta = rnorm(p_z)

  y = z %*% theta + ftrue + sigma * rnorm(n)
  y_test = ztest %*% theta + ftest + sigma * rnorm(nt)
}


#######################################################################
# XBART
categ <- function(z, j) {
  q = as.numeric(quantile(x[, j], seq(0, 1, length.out = 100)))
  output = findInterval(z, c(q, + Inf))
  return(output)
}

print("True theta is \n")
print(theta)

### prior of theta
theta_cov = diag(p_z) * 1
theta_mu = rep(0, p_z)
# theta_mu = theta

params = get_XBART_params(y)
time = proc.time()
fit = XBART.mix(as.matrix(y), as.matrix(x), as.matrix(z), as.matrix(xtest), as.matrix(ztest), as.matrix(theta_mu), as.matrix(theta_cov), p_categorical = dcat,
            params$num_trees, params$num_sweeps, params$max_depth,
            params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
            mtry = params$mtry, verbose = verbose,
            num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100, no_split_penality = params$no_split_penality)

################################
# two ways to predict on testing set

# 1. set xtest as input to main fitting function
fhat.1 = apply(fit$yhats_test[, params$burnin:params$num_sweeps], 1, mean)
time = proc.time() - time
print(time)

# 2. a separate predict function
pred = predict(fit, xtest, ztest)
pred = rowMeans(pred[, params$burnin:params$num_sweeps])
time_XBART = round(time[3], 3)




#######################################################################
# print
xbart_rmse = sqrt(mean((fhat.1 - ftest) ^ 2))
print(paste("rmse of fit xbart: ", round(xbart_rmse, digits = 4)))

par(mfrow = c(1,2))
plot(ftest, fhat.1, pch = 20, col = 'slategray', main = "Predict vs true")
abline(0,1)
legend("topleft", c("XBART"), col = c("slategray"), pch = c(20, 20))

plot(colMeans(fit$theta), theta, main = "estimation vs true theta")
abline(0,1)

