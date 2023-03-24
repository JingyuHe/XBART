# Problem description -----------------------------------------------------
# Test performance on xbcf-rd

## Setup
set.seed(000)
library(XBART)
### DGPfunction(x) return(single_index(x)) #  + 1/(1+exp(-5*xf))
mu <- function(W, X){return(0.1 * rowSums(W) + 1/(1+exp(-5*X)))} 
tau <- function(W, X) return(sin(mu(W, X)) + 1) # make sure the treatment effect is non-zero

n       <- 2000
p       <- 2
c       <- 0 # Cutoff
h_overlap       <- 0.1 # overlap bandwidth 
h_test <- 0.2

## Data
w <- matrix(rnorm(n*p), n, p)
x <- rnorm(n,sd=.5)
z <- x >= c
y <- mu(w, x) + tau(w, x)*z + rnorm(n, 0, 0.2)

## XBCF
num_sweeps = 100
burnin = 20
fit.XBCFrd <- XBCF.rd(y, w, x, c, pcat_con = 0, pcat_mod = 0,
                    num_trees_mod = 40, num_trees_con = 20, num_sweeps = num_sweeps, burnin = burnin, Nmin = 20)

ntest <- 100
xtest <- rnorm(ntest, sd = 0.5)
xtest <- sort(xtest)
wtest <- matrix(rep(0, ntest*p), ntest, p)
tau.test <- tau(wtest, xtest)
ytest <- mu(wtest, xtest) + tau.test*(xtest>=c)

data <- list(y = ytest, W = wtest, X = xtest, c = c, Wtr = w, Xtr = x)
# test_xbcf_rd(fit.XBCFrd, data, 0.01, mean(tau.test))

# Make predictions on the test data
tau.prior = var(y) / (fit.XBCFrd$model_params$n_trees_con = fit.XBCFrd$model_params$n_trees_mod)
pred.XBCFrd <- predict.XBCFrd(fit.XBCFrd, W = wtest, X = xtest, Wtr = w, Xtr = x, theta = 2, tau = 0.001)

# Check yhats
rmse.yhats <- sqrt(mean((data$y - pred.XBCFrd$yhats.adj.mean)^2))
print(paste("RMSE on yhats ", round(rmse.yhats, 3), sep = ""))

# Check tauhats in the test bandwidth
test.ind <- (xtest <= c+h_test) & (xtest >= c-h_test)
ate.hat <- colMeans(pred.XBCFrd$tau.adj[test.ind, ])[(burnin + 1):num_sweeps]
expected_ate <- mean(tau.test[test.ind])
rmse.ate <- sqrt(mean((ate.hat - expected_ate)^2))
print(paste("RMSE on ATE ", round(rmse.ate, 3), sep = ""))


par(mfrow = c(1, 2))
tau.hat <- pred.XBCFrd$tau.adj.mean
plot(xtest, tau.test, ylim = range(tau.test, tau.hat))
points(xtest, tau.hat, col = 'blue')
# legend("topleft", legend = c("True", "XBCF-GP"), col = c("black", "blue"), pch = 1)

# y.hat <- rowMeans(xbcf.fit$tauhats)*z + rowMeans(xbcf.fit$muhats)
y.hat <- pred.XBCFrd$yhats.adj.mean
plot(xtest, ytest, ylim = range(ytest, y.hat))
points(xtest, y.hat, col = 'blue')
legend("topleft", legend = c("True", "XBCF-GP"), col = c("black", "blue"), pch = 1)


