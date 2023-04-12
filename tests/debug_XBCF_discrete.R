###############################################################################
# Debugging script for binary treatment XBCF
###############################################################################

# Load libraries
library(XBART)
library(dbarts)

#### 1. DATA GENERATING PROCESS
n <- 500
# set.seed(1)

# Covariates
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.2)
x3 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
x4 <- rnorm(n)
x5 <- rbinom(n, 1, 0.7)
x <- cbind(x1, x2, x3, x4, x5)

# Treatment effects
tau <- 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)

## Prognostic function
mu <- function(x) {
    lev <- c(-0.5, 0.75, 0)
    result <- 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x3]
    return(result)
}

# Propensity score and treatment assignment
pi <- pnorm(-0.5 + mu(x) - x[, 2] + 0. * x[, 4], 0, 3)
z <- rbinom(n, 1, pi)
# hist(pi,100)

# Outcome variable
mu_x <- mu(x)
Ey <- mu_x + tau * z
sig <- 0.25 * sd(Ey)
y <- Ey + sig * rnorm(n)

# Use true pi as pihat for convenience
pihat <- pi

# Convert the data into the format needed to fit XBCF
x_original <- x
x <- data.frame(x)
x[, 3] <- as.factor(x[, 3])
x <- makeModelMatrixFromDataFrame(data.frame(x))
x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])

# Add pihat to the prognostic term matrix
x1 <- cbind(pihat, x)

# Split X into prognostic and treatment function covariates
x_con = x
x_mod <- x

#### 2. Model Fitting

# XBCF Discrete
t1 = proc.time()
xbcf.fit.xb <- XBART::XBCF.discrete(y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat, p_categorical_con = 5, p_categorical_mod = 5, num_sweeps = 60, burnin = 30)
t1 = proc.time() - t1

pred <- predict(xbcf.fit.xb, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = 30)

tauhats <- pred$tau.adj.mean

# Evaluate RMSE and runtime
print(paste0("xbcf binary RMSE: ", sqrt(mean((tauhats - tau)^2))))
print(paste0("xbcf binary runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))

# Inspect sigma0 and sigma1 samples
rowMeans(xbcf.fit.xb$sigma0)
rowMeans(xbcf.fit.xb$sigma1)

# # Plot results
# plot(tau, tauhats)
# abline(0, 1)
