# Problem description -----------------------------------------------------
# Test performance on xbcf-rd

## Setup
set.seed(1234)
library(XBART)
### DGP
mu <- function(W, X){return(0.1 * rowSums(W) + 1/(1+exp(-5*X)))} 
tau <- function(W, X) return( sin(mu(W, X)) +1) # make sure the treatment effect is non-zero

n       <- 5000
p       <- 2
c       <- 0 # Cutoff

h_test <- 0.25

## Data
w <- matrix(rnorm(n*p), n, p)
x <- rnorm(n,sd=.5)
z <- x >= c
y <- mu(w, x) + tau(w, x)*z + rnorm(n, 0, 0.2)

## XBCF
num_sweeps = 100
burnin =10
fit.XBCFrd <- XBCF.rd(y, w, x, c, Owidth = 0.03, Omin = 50, Opct = 0.95, pcat_con = 0, pcat_mod = 0,
                      num_trees_mod = 5, num_trees_con = 20, num_cutpoints = n, num_sweeps = num_sweeps, burnin = burnin, Nmin = 20)

# Test set generation
ntest <- 100
xtest <- rnorm(ntest, sd = 0.5)
xtest <- sort(xtest)
wtest <- matrix(rep(0, ntest*p), ntest, p)
tau.test <- tau(wtest, xtest)
ytest <- mu(wtest, xtest) + tau.test*(xtest>=c)
data <- list(y = ytest, W = wtest, X = xtest, c = c, Wtr = w, Xtr = x)

## Case 1: Predict tau(X) without GP
# Make predictions on the test data
tau.prior = var(y) / (fit.XBCFrd$model_params$n_trees_con = fit.XBCFrd$model_params$n_trees_mod)
pred.XBCFrd <- predict.XBCFrd(fit.XBCFrd, W = wtest, X = xtest, Wtr = w, Xtr = x)

# Check yhats
y.hat <- pred.XBCFrd$yhats.adj.mean
rmse.yhats <- sqrt(mean((data$y - y.hat)^2))
print(paste("XBCF (non-GP) RMSE on yhats ", round(rmse.yhats, 3), sep = ""))

# Check tauhats in the test bandwidth
test.ind <- (xtest <= c+h_test) & (xtest >= c-h_test)
cate.est <- rowMeans(pred.XBCFrd$tau.adj[test.ind, (burnin + 1):num_sweeps])
rmse.cate <- sqrt(mean((cate.est - tau.test[test.ind])^2))
print(paste("XBCF (non-GP) RMSE on CATE ", round(rmse.cate, 3), sep = ""))
tau.hat <- pred.XBCFrd$tau.adj.mean

## Case 2: Predict tau(X) with GP
# Make predictions on the test data
pred.XBCFrdgp <- predict.XBCFrdgp(fit.XBCFrd, W = wtest, X = xtest, Wtr = w, Xtr = x, theta = 0.03, tau = 0.1)

# Check yhats
y.hat.gp <- pred.XBCFrdgp$yhats.adj.mean
rmse.yhats.gp <- sqrt(mean((data$y - y.hat.gp)^2))
print(paste("XBCF (GP) RMSE on yhats ", round(rmse.yhats.gp, 3), sep = ""))

# Check tauhats in the test bandwidth
cate.est.gp <- rowMeans(pred.XBCFrdgp$tau.adj[test.ind, (burnin + 1):num_sweeps])
rmse.cate.gp <- sqrt(mean((cate.est.gp - tau.test[test.ind])^2))
print(paste("XBCF (GP) RMSE on CATE ", round(rmse.cate.gp, 3), sep = ""))
tau.hat.gp <- pred.XBCFrdgp$tau.adj.mean

# Plot results
par(mfrow = c(2, 2))
first_col_ylim = range(tau.test, tau.hat, tau.hat.gp)
second_col_ylim = range(ytest, y.hat, y.hat.gp)

plot(xtest, tau.test, xlim=c(-0.25,0.25),pch=20,type='l',main="Tauhat (XBCF no GP)", ylim = first_col_ylim)
points(xtest, tau.hat, col = 'blue',pch=20)
abline(v=c,col='red')

plot(xtest, ytest, main="Yhat (XBCF no GP)", ylim = second_col_ylim)
points(xtest, y.hat, col = 'blue')
legend("topleft", legend = c("True", "XBCF"), col = c("black", "blue"), pch = 1)

plot(xtest, tau.test, xlim=c(-0.25,0.25),pch=20,type='l',main="Tauhat (XBCF GP)", ylim = first_col_ylim)
points(xtest, tau.hat.gp, col = 'blue',pch=20)
abline(v=c,col='red')

plot(xtest, ytest, main="Yhat (XBCF GP)", ylim = second_col_ylim)
points(xtest, y.hat.gp, col = 'blue')
legend("topleft", legend = c("True", "XBCF-GP"), col = c("black", "blue"), pch = 1)
