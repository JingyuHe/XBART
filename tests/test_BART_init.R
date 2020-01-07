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
  XBART_params$num_cutpoints = 100;
  # number of adaptive cutpoints
  XBART_params$tau = var(y) / num_tress # prior variance of mu (leaf parameter)
  return(XBART_params)
}


# data generating process
# it can be a string of linear, singleindex, tripoly or max
type = "singleindex"
noise_ratio = 1
rep = 5

cover = matrix(0, rep, 3)
len = matrix(0, rep, 3)
running_time = matrix(0, rep, 3)
rmse = matrix(0, rep, 3)

#######################################################################
library(XBART)
library(BART)

set.seed(100)
new_data = TRUE # generate new data
run_dbarts = FALSE # run dbarts
run_xgboost = FALSE # run xgboost
run_lightgbm = FALSE # run lightgbm
parl = TRUE # parallel computing


small_case = TRUE # run simulation on small data set
verbose = FALSE # print the progress on screen


if (small_case) {
  n = 10000 # size of training set
  nt = 5000 # size of testing set
  d = 30 # number of TOTAL variables
  dcat = 0 # number of categorical variables
  # must be d >= dcat
  # (X_continuous, X_categorical), 10 and 10 for each case, 20 in total
} else {
  n = 1000000
  nt = 10000
  d = 50
  dcat = 0
}





for(kk in 1:rep){


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
    # sin(rowSums(x[, 3:4] ^ 2)) + sin(rowSums(x[, 1:2] ^ 2)) + (x[, 15] + x[, 14]) ^ 2 * (x[, 1] + x[, 2] ^ 2) / (3 + x[, 3] + x[, 14] ^ 2)
    #rowSums(x[,1:30]^2)
    #pmax(x[,1]*x[,2], abs(x[,3])*(x[,10]>x[,15])+abs(x[,4])*(x[,10]<=x[,15]))
    output = 0

    if(type == "linear"){
      for(i in 1:d){
        output = output + x[,i] * (-2 + 4 * (i -1) / (d - 1))
      }
    }else if(type == "singleindex"){
      a = 0
      for(i in 1:10){
        g = -1.5 + (i - 1) / 3
        a = a + (x[,i] - g)^2
      }
      output = 10 * sqrt(a) + sin(5 * a)
    }else if(type == "tripoly"){
      output = 5 * sin(3 * x[,1]) + 2 * x[,2]^2 + 3 * x[,3] * x[,4]
    }else if(type == "max"){
      output = rep(0, dim(x)[1])
      for(i in 1:(dim(x)[1])){
        output[i] = max(max(x[i, 1], x[i, 2]), x[i, 3])
      }
    }
    return(output)
  }

  # to test if ties cause a crash in continuous variables
  # x[, 1] = round(x[, 1], 4)
  #xtest[,1] = round(xtest[,1],2)
  ftrue = f(x)
  ftest = f(xtest)
  sigma = noise_ratio * sd(ftrue)

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




params = get_XBART_params(y)
time = proc.time()









# XBART
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), p_categorical = dcat, 
            params$num_trees, params$num_sweeps, params$max_depth,
            params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1,
            mtry = params$mtry, verbose = TRUE,
            num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100, no_split_penality = params$no_split_penality)

################################
# two ways to predict on testing set

# 1. set xtest as input to main fitting function
fhat.1 = apply(fit$yhats_test[, params$burnin:params$num_sweeps], 1, mean)
time = proc.time() - time
print(time[3])

# 2. a separate predict function
pred = predict(fit, xtest)
pred = rowMeans(pred[, params$burnin:params$num_sweeps])

time_XBART = round(time[3], 3)

pred2 = predict(fit, xtest) 
pred2 = rowMeans(pred2[, params$burnin:params$num_sweeps])
stopifnot(pred == pred2)



#####
# bart with default initialization
time = proc.time()
fit_bart = wbart(x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, ndpost = 100 * (params$num_sweeps - params$burnin), nskip = 1000)
time = proc.time() - time
time_BART = round(time[3], 3)

pred_bart = colMeans(predict(fit_bart, xtest))


# # bart with XBART initialization
# fit_bart2 = wbart_ini(treedraws = fit$treedraws, x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, nskip = 0, ndpost = 100, sigest = mean(fit$sigma))


# pred_bart_ini = colMeans(predict(fit_bart2, xtest))


# xbart_rmse = sqrt(mean((fhat.1 - ftest) ^ 2))
# bart_rmse = sqrt(mean((pred_bart - ftest)^2))
# bart_ini_rmse = sqrt(mean((pred_bart_ini - ftest)^2))


# xbart_rmse
# bart_rmse
# bart_ini_rmse





#######################################################################
# Calculate coverage
#######################################################################

# coverage of the real average
draw_BART_XBART = c()

time_warm_start_all = rep(0, length(params$burnin:params$num_sweeps))
for(i in params$burnin:params$num_sweeps){
  # bart with XBART initialization
  cat("------------- i ", i , "\n")
  set.seed(1)
  time = proc.time()
  fit_bart2 = wbart_ini(treedraws = fit$treedraws[i], x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, nskip = 0, ndpost = 100)
  time = proc.time() - time

  draw_BART_XBART = rbind(draw_BART_XBART, fit_bart2$yhat.test)

  time_warm_start_all[i - params$burnin + 1] = time[3]
}


# #######################################################################
# # print
xbart_rmse = sqrt(mean((fhat.1 - ftest) ^ 2))
bart_rmse = sqrt(mean((pred_bart - ftest)^2))
bart_ini_rmse = sqrt(mean((colMeans(draw_BART_XBART) - ftest)^2))


xbart_rmse
bart_rmse
bart_ini_rmse





coverage = c(0,0,0)

length = matrix(0, nt, 3)

for(i in 1:nt){
  lower = quantile(fit$yhats_test[i, params$burnin:params$num_sweeps], 0.025)
  higher = quantile(fit$yhats_test[i, params$burnin:params$num_sweeps], 0.975)
  if(ftest[i] < higher && ftest[i] > lower){
    coverage[1] = coverage[1] + 1
  }
  length[i,1] = higher - lower

  lower = quantile(fit_bart$yhat.test[,i], 0.025)
  higher = quantile(fit_bart$yhat.test[,i], 0.975)
  if(ftest[i] < higher && ftest[i] > lower){
    coverage[2] = coverage[2] + 1
  }
  length[i,2] = higher - lower

  lower = quantile(draw_BART_XBART[,i], 0.025)
  higher = quantile(draw_BART_XBART[,i], 0.975)
  if(ftest[i] < higher && ftest[i] > lower){
    coverage[3] = coverage[3] + 1
  }
  length[i,3] = higher - lower

}

cover[kk, ] = coverage / nt
len[kk, ] = colMeans(length)
running_time[kk, ] = c(time_XBART, time_BART, mean(time_warm_start_all))
rmse[kk, ] = c(xbart_rmse, bart_rmse, bart_ini_rmse)
}

results = rbind(colMeans(cover), colMeans(len), colMeans(running_time), colMeans(rmse))
colnames(results) = c("XBART", "BART", "warm start")
rownames(results) = c("coverage", "interval length", "running time", "RMSE")

results = round(results, 4)

print(results)

