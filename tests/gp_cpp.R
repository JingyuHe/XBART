func = 'single_index'


set.seed(230)

library(MASS)
library(XBART)
library(BART) #Jingyu's version

n = 5000
d = 5
kappa = 1


# functions ---------------------------------------------------------------

linear = function(x){
  d = ncol(x)
  beta = -2 + 4*(c(1:d) - 1) / (d-1)
  return(x %*% beta)
}

single_index = function(x){
  if (ncol(x) > 10) x = x[, 1:10] # discard extra columns
  d = ncol(x)
  gamma = -1.5 + (c(1:d) - 1) /3
  a = apply(x, 1, function(x) sum((x - gamma)^2))
  f = 10 * sqrt(a) + sin(5*a)
  return(f)
}

trig_poly <- function(x){
  f <- -2 +  x[,1]^2 
  return(f)
}

max_func <- function(x){
  f <- apply(x, 1, function(x) return(max(x[1:3])))
  return(f)
}

f_true <- function(x, func){
  if(func == 'linear') return(linear(x))
  if(func == 'single_index') return(single_index(x))
  if(func == 'trig_poly') return(trig_poly(x))
  if(func == 'max') return(max_func(x))
}



gaussian_kernel <- function(x, y, theta,tau){
  # exp(-sum(theta * (x_i - x_j)^2))
  stopifnot(length(x) == length(y))
  return(tau*exp(-sum(theta * (x - y)^2)))
}

get_covariance <- function(X, theta, tau){
  cov = matrix(NA, nrow(X), nrow(X))
  for (i in 1:nrow(X)){
    for (j in i:nrow(X)){
      cov[i, j] = cov[j, i] = gaussian_kernel(X[i,], X[j,], theta, tau)
    }
  }
  return(cov)
}

rel_kernel <- function(x, y, theta, tau, range){
  # exp(-sum(theta * (x_i - x_j)^2))
  stopifnot(length(x) == length(y))
  return(tau*exp(-sum(theta * abs(x - y) / range)))
}

get_rel_covariance <- function(X, theta, tau, range){
  cov = matrix(NA, nrow(X), nrow(X))
  for (i in 1:nrow(X)){
    for (j in i:nrow(X)){
      cov[i, j] = cov[j, i] = rel_kernel(X[i,], X[j,], theta, tau, range)
    }
  }
  return(cov)
}


# get data ----------------------------------------------------------------

x = matrix(rnorm(n * d), n, d)
f = f_true(x, func)
sigma = 0.1*sqrt(kappa^2 * var(f))
y = f + rnorm(n, 0, sigma)

x_range <- sapply(1:d, function(i, x) max(x[,i]) - min(x[,i]), x)

# # replicate test set
nt = 30
nrep = 1 # 100 replicates per dp
xt = matrix(0, nt * nrep, d) # fix all variables but one
x1 = seq(min(x[,1]) - 4, max(x[,1]) + 4, length.out = nt)
xt[,1] = sapply(x1, rep, nrep) # repeat each dp nrep times

# random test set
# nt = 1000
# xt = matrix(rnorm(nt * d), nt, d)
# x1 = xt[,1]
# ft = f_true(xt, func)
# yt = ft + rnorm(nt, 0, sigma)


# uniform test set
# nt = 1000
# xt = matrix(runif(nt * d, round(min(x)) - 5, round(max(x))) + 5, nt, d) # extropolation
# x1 = xt[,1]
# ft = f_true(xt, func)
# yt = ft + rnorm(nt, 0, sigma)



ft = f_true(xt, func)
yt = ft + rnorm(nt, 0, sigma)



# train -------------------------------------------------------
tau = var(y)/10
n_trees = 10
fit <- XBART(y=matrix(y),  X=x, Xtest=xt, num_trees=n_trees, Nmin = 10,num_sweeps=200, burnin = 15, tau = tau, sampling_tau = TRUE)

gp_pred <- predict.gp(fit, as.matrix(y), x, xt, theta = 1, tau = 1, p_categorical = 0)


gp_yhat <- t(apply(gp_pred, 1, function(x) rnorm(length(x), x, fit$sigma[10,])))

gp.upper <- apply(gp_yhat, 1, quantile, 0.975, na.rm = TRUE)
gp.lower <- apply(gp_yhat, 1, quantile, 0.025, na.rm = TRUE)
coverage.gp <- (yt <= gp.upper & yt >= gp.lower)

print(paste("gp rmse = ", sqrt(mean((yt - rowMeans(gp_yhat))^2, na.rm=TRUE))))
print(paste("gp coverage = ", mean(coverage.gp)))
print(paste("gp interval = ", mean(gp.upper - gp.lower)))

x1 <- xt[,1]
plot(x1, yt, pch = 19, cex = 0.5,ylim=c(-50,100))

lines(x1, ft)
# lines(x1, xbart.upper, col = 2)
# lines(x1, xbart.lower, col = 2)
lines(x1, gp.upper, col = 3)
lines(x1, gp.lower, col = 3)
abline(v = min(x[,1]), col = 4)
abline(v = max(x[,1]), col = 4)
legend('bottomright', legend = c('ftrue', 'xbart', 'gp'), col = c(1:3), lty = 1)

