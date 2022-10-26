# simple demonstration of XBCF with default parameters
library(XBART)
library(dbarts)

#### 1. DATA GENERATION PROCESS
n <- 5000 # number of observations
# set seed here
# set.seed(1)

# generate dcovariates
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.2)
x3 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
x4 <- rnorm(n)
x5 <- rbinom(n, 1, 0.7)
x <- cbind(x1, x2, x3, x4, x5)

# define treatment effects
tau <- 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)

## define prognostic function (RIC)
mu <- function(x) {
  lev <- c(-0.5, 0.75, 0)
  result <- 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x3]
  return(result)
}

# compute propensity scores and treatment assignment
pi <- pnorm(-0.5 + mu(x) - x[, 2] + 0. * x[, 4], 0, 3)
# hist(pi,100)
z <- rbinom(n, 1, pi)

# x1 = rnorm(n)
# x2 = rnorm(n)
# x3 = rnorm(n)
# x4 = rnorm(n)
# x5 = rnorm(n)

# x = cbind(x1, x2, x3, x4, x5)

# tau <- 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)
# mu <- function(x) {
#   lev <- c(-0.5, 0.75, 0)
#   result <- 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + x[, 3]
#   return(result)
# }
# pi <- pnorm(-0.5 + mu(x) - x[, 2] + 0. * x[, 4], 0, 3)
# z <- rbinom(n, 1, pi)


# generate outcome variable
Ey <- mu(x) + tau * z
sig <- 0.25 * sd(Ey)
y <- Ey + sig * rnorm(n)

# If you didn't know pi, you would estimate it here
pihat <- pi

# matrix prep
x <- data.frame(x)
x[, 3] <- as.factor(x[, 3])
x <- makeModelMatrixFromDataFrame(data.frame(x))
x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])

# add pihat to the prognostic term matrix
x1 <- cbind(pihat, x)

x_con = x
x_mod = x


#### 2. XBCF

# run XBCF
t1 <- proc.time()
xbcf.fit <- XBCF::XBCF(y = y, z = z, x_con = x_con, x_mod = x_mod, pihat = pihat, pcat_con = 5, pcat_mod = 5, num_sweeps = 60, burnin = 30)
xbcf.fit <- XBART::XBCF.discrete(y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat, p_categorical_con = 5, p_categorical_mod = 5, num_sweeps = 1, burnin = 0, verbose = TRUE, sampling_tau = FALSE, num_trees_con = 5, num_trees_mod = 5, a_scaling = FALSE, b_scaling = FALSE)

t1 <- proc.time() - t1

# get treatment individual-level estimates
tauhats <- getTaus(xbcf.fit)

# main model parameters can be retrieved below
# print(xbcf.fit$model_params)

# compare results to inference
plot(tau, tauhats)
abline(0, 1)
print(paste0("xbcf RMSE: ", sqrt(mean((tauhats - tau)^2))))
print(paste0("xbcf runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))
