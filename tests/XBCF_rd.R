# Problem description -----------------------------------------------------
# Test performance on xbcf-rd

## Setup
set.seed(000)
library(XBART)
### DGPfunction(x) return(single_index(x)) #  + 1/(1+exp(-5*xf))
# mu <- function(W, X){return(0.1 * rowSums(W) + 1/(1+exp(-5*X)))} 
# tau <- function(W, X) return( sin(mu(W, X)) + 1) # make sure the treatment effect is non-zero

# 1d
mu <- function(W, X){return(1/(1+exp(-5*X)))}
tau <- function(W, X) return( sin(mu(W, X)) + 1) # make sure the treatment effect is non-zero


n       <- 2000
p       <- 0
c       <- 0 # Cutoff
h_overlap       <- 0.1 # overlap bandwidth 
h_test <- 0.2

## Data
w <- matrix(rnorm(n*p), n, p)
x <- rnorm(n,sd=.5)
z <- x >= c
y <- mu(w, x) + tau(w, x)*z + rnorm(n, 0, 0.1)

## XBCF
num_sweeps = 12
burnin = 10
fit.XBCFrd <- XBCF.rd(y, w, x, c, Owidth = 0.1, Omin = 10, Opct = 0.9, pcat_con = 0, pcat_mod = 0,
                    num_trees_mod = 2, num_trees_con = 20, num_cutpoints = n, num_sweeps = num_sweeps, burnin = burnin, Nmin = 20)

ntest <- 100
# xtest <- rnorm(ntest, sd = 0.5)
xtest <- runif(ntest, -h_test, h_test)
xtest <- sort(xtest)
wtest <- matrix(rep(0, ntest*p), ntest, p)
tau.test <- tau(wtest, xtest)
ytest <- mu(wtest, xtest) + tau.test*(xtest>=c)

data <- list(y = ytest, W = wtest, X = xtest, c = c, Wtr = w, Xtr = x)
# test_xbcf_rd(fit.XBCFrd, data, 0.01, mean(tau.test))

# Make predictions on the test data
tau.prior = var(y) / (fit.XBCFrd$model_params$n_trees_con = fit.XBCFrd$model_params$n_trees_mod)

pred.XBCFrd <- predict.XBCFrd(fit.XBCFrd, W = wtest, X = xtest)
pred.XBCFrdgp <- predict.XBCFrdgp(fit.XBCFrd, W = wtest, X = xtest, Wtr = w, Xtr = x, theta = 1, tau = 0.1)

# Check yhats
rmse.yhats <- sqrt(mean((data$y - pred.XBCFrd$yhats.adj.mean)^2))
print(paste("RMSE on yhats ", round(rmse.yhats, 3), sep = ""))

# Check tauhats in the test bandwidth
test.ind <- (xtest <= c+h_test) & (xtest >= c-h_test)
expected_ate <- mean(tau.test[test.ind])
ate.hat <- colMeans(pred.XBCFrd$tau.adj[test.ind, ])[(burnin + 1):num_sweeps]
rmse.ate <- sqrt(mean((ate.hat - expected_ate)^2))
print(paste("XBCF RMSE on ATE ", round(rmse.ate, 3), sep = ""))

ate.hat.gp <- colMeans(pred.XBCFrdgp$tau.adj[test.ind, ])[(burnin + 1):num_sweeps]
rmse.ate.gp <- sqrt(mean((ate.hat.gp - expected_ate)^2))
print(paste("XBCF-GP RMSE on ATE ", round(rmse.ate.gp, 3), sep = ""))


par(mfrow = c(2, 2))
tau.hat <- pred.XBCFrd$tau.adj.mean
tau.hat.gp <- pred.XBCFrdgp$tau.adj.mean
plot(xtest, tau.test, ylim = range(tau.test, tau.hat, tau.hat.gp), main = 'XBCF')
points(xtest, tau.hat, col = 'blue')
legend("topleft", legend = c("True", "Estimate"), col = c("black", "blue"), pch = 1)

plot(xtest, tau.test, ylim = range(tau.test, tau.hat.gp, tau.hat), main = 'XBCF-GP')
points(xtest, tau.hat.gp, col = 'blue')
abline(v = -0.1)
abline(v = 0.1)

y.hat <- pred.XBCFrd$yhats.adj.mean
plot(xtest, ytest, ylim = range(ytest, y.hat), main = 'yhat')
points(xtest, y.hat, col = 'blue')
# legend("topleft", legend = c("True", "XBCF-GP"), col = c("black", "blue"), pch = 1)

mu.test <- mu(wtest, xtest)
mu.hat <- pred.XBCFrd$mu.adj.mean
plot(xtest, ytest - mu.test, ylim = range(ytest - mu.test, y.hat - mu.hat), main = 'y - mu')
points(xtest, y.hat - mu.hat, col = 'blue')

