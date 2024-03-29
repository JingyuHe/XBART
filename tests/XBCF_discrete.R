# simple demonstration of XBCF with default parameters
library(XBART)
library(dbarts)

reps <- 1
rmse.stats <- matrix(NA, nrow = reps, ncol = 2)
colnames(rmse.stats) <- c("xbcf::xbcf", "xbart::xbcf")

for(i in c(1:reps)) {
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
  
  # generate outcome variable
  Ey <- mu(x) + tau * z
  sig <- 0.25 * sd(Ey)
  y <- Ey + sig * rnorm(n)
  
  # If you didn't know pi, you would estimate it here
  pihat <- pi
  
  # store mu values before processing input matrix
  muvec <- mu(x)
  
  # matrix prep
  x <- data.frame(x)
  x[, 3] <- as.factor(x[, 3])
  x <- makeModelMatrixFromDataFrame(data.frame(x))
  x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])
  
  # add pihat to the prognostic term matrix
  x1 <- cbind(pihat, x)
  
  x_con = x
  x_mod <- x
  
  
  #### 2. XBCF
  
  # run XBCF (XBCF repo)
  t1 <- proc.time()
  xbcf.fit <- XBCF::XBCF(y = y, z = z, x_con = x_con, x_mod = x_mod, pihat = pihat, pcat_con = 5, pcat_mod = 5, num_sweeps = 60, burnin = 30)
  t1 <- proc.time() - t1
  # get treatment individual-level estimates
  tauhats <- XBCF::getTaus(xbcf.fit)
  muhats <- XBCF::getMus(xbcf.fit)
  
  # compare results to inference
  par(mfrow = c(1, 2))
  plot(tau, tauhats)
  abline(0, 1)
  plot(muvec, muhats)
  abline(0, 1)
  print(paste0("xbcf RMSE: ", sqrt(mean((tauhats - tau)^2))))
  print(paste0("xbcf runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))

  rmse.stats[i,1] <- sqrt(mean((tauhats - tau)^2))
    
  # run XBCF (XBART repo)
  t2 = proc.time()
  xbcf.fit.xb <- XBART::XBCF.discrete(y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat, p_categorical_con = 5, p_categorical_mod = 5, num_sweeps = 60, burnin = 30)
  t2 = proc.time() - t2
  
  pred <- predict(xbcf.fit.xb, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = 30)
  
  # taus2 <- pred$tau.adj
  # tauhats2 = rowMeans(taus2)
  tauhats2 <- pred$tau.adj.mean
  muhats2 <- pred$mu.adj.mean
  
  plot(tau, tauhats2)
  abline(0, 1)
  plot(muvec, muhats2)
  abline(0, 1)
  print(paste0("xbcf binary RMSE: ", sqrt(mean((tauhats2 - tau)^2))))
  print(paste0("xbcf binary runtime: ", round(as.list(t2)$elapsed, 2), " seconds"))
  rmse.stats[i,2] <- sqrt(mean((tauhats2 - tau)^2))
  # check predicted outcomes
  # plot(y,rowMeans(pred$yhats.adj[,30:60]))
}


# main model parameters can be retrieved below
# print(xbcf.fit$model_params)
