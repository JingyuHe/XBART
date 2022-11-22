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
  
  num_sweeps <-60
  burnin <- 30
  # run XBCF heteroskedastic
  t1 = proc.time()
  fit.hsk <- XBCF.discrete.heterosk(y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat, 
                                    p_categorical_con = 5, p_categorical_mod = 5, 
                                    num_trees_con = 5, num_trees_mod = 5,
                                    num_sweeps = num_sweeps, burnin = burnin)
  t1 = proc.time() - t1
  
  pred <- predict.XBCFdiscreteHeterosk(fit.hsk, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
  tauhats <- pred$tau.adj.mean
  
  # compare results to inference
  par(mfrow = c(1, 2))
  plot(tau, tauhats)
  abline(0, 1)
  print(paste0("xbcf hsk RMSE: ", sqrt(mean((tauhats - tau)^2))))
  print(paste0("xbcf hsk runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))
  
  rmse.stats[i,1] <- sqrt(mean((tauhats - tau)^2))
  
  # run XBCF homoskedastic
  t2 = proc.time()
  fit <- XBCF.discrete(y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat, 
                       p_categorical_con = 5, p_categorical_mod = 5, 
                       num_sweeps = num_sweeps, burnin = burnin, 
                       num_trees_con = 5, num_trees_mod = 5,
                       a_scaling = FALSE, b_scaling = FALSE)
  t2 = proc.time() - t2
  
  pred2 <- predict(fit, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
  tauhats2 <- pred2$tau.adj.mean
  
  
  plot(tau, tauhats2)
  abline(0, 1)
  print(paste0("xbcf RMSE: ", sqrt(mean((tauhats2 - tau)^2))))
  print(paste0("xbcf runtime: ", round(as.list(t2)$elapsed, 2), " seconds"))
  rmse.stats[i,2] <- sqrt(mean((tauhats2 - tau)^2))
  # check predicted outcomes
  # plot(y,rowMeans(pred$yhats.adj[,30:60]))
}


# main model parameters can be retrieved below
# print(xbcf.fit$model_params)
