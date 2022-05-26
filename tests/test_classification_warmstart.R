###################################################
# This script shows initialize BART using XBART trees
# classification case
# use the customized BART package https://github.com/jingyuhe/BART
###################################################



library(XBART)
library(BART)


plotROC <- function(pihat, ytrue, add = FALSE, col = "steelblue") {
    thresh <- sort(pihat)
    N <- length(pihat)
    yhat <- sapply(1:N, function(a) as.double(pihat >= thresh[a]))
    tpr <- sapply(1:N, function(a) length(which(ytrue == 1 & yhat[, a] == 1)) / sum(ytrue == 1))
    fpr <- sapply(1:N, function(a) length(which(ytrue == 0 & yhat[, a] == 1)) / sum(ytrue == 0))
    if (add == FALSE) {
        plot(fpr, tpr, pch = 20, cex = 0.8, col = col, bty = "n", type = "b")
        abline(a = 0, b = 1, lty = 2)
    } else {
        points(fpr, tpr, pch = 20, cex = 0.8, col = col, bty = "n", type = "b")
    }
}

get_entropy <- function(nclass) {
    pi <- c(0.99, rep(0.01 / (nclass - 1), nclass - 1))
    pi <- pi / sum(pi)
    return(sum(-pi * log(pi)))
}


#####################
# simulation parameters
n <- 3000 # training size
nt <- 1000 # testing size
p <- 6 # number of X variables
p_cat <- 0 # number of categorical X variables
k <- 6 # number of classes
lam <- matrix(0, n, k)
lamt <- matrix(0, nt, k)


#####################
# simulate data
K <- matrix(rnorm(3 * p), p, 3)
X_train <- t(K %*% matrix(rnorm(3 * n), 3, n))
X_test <- t(K %*% matrix(rnorm(3 * nt), 3, nt))
X_train <- pnorm(X_train)
X_test <- pnorm(X_test)
X_train <- cbind(X_train, matrix(rbinom(n * p_cat, 1, 0.5), nrow = n))
X_test <- cbind(X_test, matrix(rbinom(nt * p_cat, 1, 0.5), nrow = nt))
lam[, 1] <- abs(3 * X_train[, 1] - X_train[, 2])
lam[, 2] <- 2
lam[, 3] <- 3 * X_train[, 3]^2
lam[, 4] <- 4 * (X_train[, 4] * X_train[, 5])
lam[, 5] <- 2 * (X_train[, 5] + X_train[, 6])
lam[, 6] <- 2 * (X_train[, 1] + X_train[, 3] - X_train[, 5])
lamt[, 1] <- abs(3 * X_test[, 1] - X_test[, 2])
lamt[, 2] <- 2
lamt[, 3] <- 3 * X_test[, 3]^2
lamt[, 4] <- 4 * (X_test[, 4] * X_test[, 5])
lamt[, 5] <- 2 * (X_test[, 5] + X_test[, 6])
lamt[, 6] <- 2 * (X_test[, 1] + X_test[, 3] - X_test[, 5])

#####################
# vary s to make the problem harder s < 1 or easier s > 2
s <- 10
pr <- exp(s * lam)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_train <- sapply(1:n, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))

pr <- exp(s * lamt)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_test <- sapply(1:nt, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))


#####################
# parameters of XBART
num_sweeps <- 20
burnin <- 5
num_trees <- 20


# fit separate tree for each class
# or all classes share the same tree structure
# for XBART
separate_tree <- FALSE

# for warm start BART
if (separate_tree) {
    tree_type <- "shared"
} else {
    tree_type <- "separate"
}


tm <- proc.time()
fit <- XBART.multinomial(y = matrix(y_train), num_class = k, X = X_train, num_trees = num_trees, num_sweeps = num_sweeps, p_categorical = p_cat, separate_tree = separate_tree, parallel = FALSE)

# initialize BART at XBART trees
# the warm start bart only draws 100 samples, without burnin, thinning = 1
fit2 <- mlbart_ini(fit$treedraws, x.train = X_train, y.train = y_train, num_class = k, x.test = X_test, type = tree_type, ntree = num_trees, ndpost = 100, nskip = 0, keepevery = 1)
tm <- proc.time() - tm


cat(paste("\n", "XBART runtime: ", round(tm["elapsed"], 3), " seconds"), "\n")


#####################
# out of sample prediction of XBART
# take average of all sweeps, discard burn-in
pred <- predict(fit, X_test)
a <- apply(pred$yhats[burnin:num_sweeps, , ], c(2, 3), median)
pred <- apply(a, 1, which.max) - 1
yhat <- apply(a, 1, which.max) - 1
cat(paste("XBART classification accuracy: ", round(mean(y_test == yhat), 3)), "\n")


#####################
# out of sample prediction of warm-start BART
# not necessary to discard burnin since they are initialized at XBART trees!
pred_warmstart <- fit2$yhat.test
a_warmstart <- apply(pred_warmstart, c(2, 3), median)
pred_warmstart <- apply(a_warmstart, 2, which.max) - 1
yhat_warmstart <- apply(a_warmstart, 2, which.max) - 1
cat(paste("warm-start BART classification accuracy: ", round(mean(y_test == yhat_warmstart), 3)), "\n")