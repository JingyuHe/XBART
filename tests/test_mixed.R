#######################################################################
# set parameters of XBART
get_XBART_params <- function(n, d, y) {
  XBART_params = list(M = 30,
                      L = 1,
                      nsweeps = 40,
                      Nmin = 1,
                      alpha = 0.95,
                      beta = 1.25,
                      mtry = 5,
                      burnin = 15)
  num_tress = XBART_params$M
  XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
  XBART_params$Ncutpoints = 50;
  XBART_params$tau = var(y) / num_tress
  return(XBART_params)
}


#######################################################################
library(XBART)
library(dbarts)



d = 20 # number of TOTAL variables
dcat = 20 # number of categorical variables

# must be d >= dcat

n = 5000 # size of training set
nt = 1000 # size of testing set

new_data = TRUE # generate new data
run_dbarts = TRUE # run dbarts
parl = FALSE # parallel computing


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
# dbarts
if (run_dbarts) {
  time = proc.time()
  fit = bart(x, y, xtest, verbose = FALSE, numcut = 100, ndpost = 1000, nskip = 500)
  time = proc.time() - time
  print(time[3])
  fhat.db = fit$yhat.test.mean
  time_dbarts = round(time[3], 3)
}

#######################################################################
# XBART
categ <- function(z, j) {
  q = as.numeric(quantile(x[, j], seq(0, 1, length.out = 100)))
  output = findInterval(z, c(q, + Inf))
  return(output)
}


params = get_XBART_params(n, d, y)
time = proc.time()
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), p_categorical = dcat,
            params$M, params$L, params$nsweeps, params$max_depth,
            params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
            mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE, draw_mu = TRUE,
            Ncutpoints = params$Ncutpoints, parallel = parl)


fhat.1 = apply(fit$yhats_test[, params$burnin:params$nsweeps], 1, mean)
time = proc.time() - time
print(time[3])

time_XBART = round(time[3], 3)

#######################################################################
# XBART, old function call, identical algorithm
if (1) {
  time = proc.time()
  fit2 = train_forest_root_std_all(as.matrix(y), as.matrix(x), as.matrix(xtest), params$M, params$L, p_categorical = dcat, params$nsweeps, params$max_depth,
                                 params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
                                 mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE, draw_mu = TRUE,
                                 Ncutpoints = params$Ncutpoints, parallel = parl)
  fhat.2 = apply(fit2$yhats_test[, params$burnin:params$nsweeps], 1, mean)
  time = proc.time() - time
  print(time[3])
}
time_XBART_old = round(time[3], 3)


#######################################################################
# print
print(paste("rmse of fit xbart: ", round(sqrt(mean((fhat.1 - ftest) ^ 2)), digits = 4)))
print(paste("rmse of fit xbart, old function call: ", round(sqrt(mean((fhat.2 - ftest) ^ 2)), digits = 4)))
print(paste("rmse of fit dbart: ", round(sqrt(mean((fhat.db - ftest) ^ 2)), digits = 4)))

print(paste("running time, dbarts", time_dbarts))
print(paste("running time, XBART", time_XBART))
print(paste("running time, XBART old function call", time_XBART_old))

plot(ftest, fhat.db, pch = 20, col = 'orange')
points(ftest, fhat.1, pch = 20, col = 'slategray')
points(ftest, fhat.2, pch = 20, col = 'cyan')
legend("topleft", c("dbarts", "XBART", "XBART.old"), col = c("orange", "slategray", "cyan"), pch = c(20, 20, 20))

