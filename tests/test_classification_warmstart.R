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
  # print(mean(tpr))
}

get_entropy <- function(nclass) {
  pi <- c(0.99, rep(0.01 / (nclass - 1), nclass - 1))
  pi <- pi / sum(pi)
  return(sum(-pi * log(pi)))
}

# seed = 10
# set.seed(seed)

#
# n = 200
# nt = 50
n <- 1000
nt <- 5000
p <- 20
p_cat <- 20
k <- 6
lam <- matrix(0, n, k)
lamt <- matrix(0, nt, k)


K <- matrix(rnorm(3 * p), p, 3)
X_train <- t(K %*% matrix(rnorm(3 * n), 3, n))
X_test <- t(K %*% matrix(rnorm(3 * nt), 3, nt))

X_train <- pnorm(X_train)
X_test <- pnorm(X_test)

# X_train = matrix(runif(n*p,-1,1), nrow=n)
# X_test = matrix(runif(nt*p,-1,1), nrow=nt)

X_train <- cbind(X_train, matrix(rbinom(n * p_cat, 1, 0.5), nrow = n))
X_test <- cbind(X_test, matrix(rbinom(nt * p_cat, 1, 0.5), nrow = nt))

# X_train = cbind(X_train, matrix(rpois(n*p_cat, 20), nrow=n))
# X_test = cbind(X_test, matrix(rpois(nt*p_cat, 20), nrow=nt))


lam[, 1] <- abs(2 * X_train[, 1] - X_train[, 2])
lam[, 2] <- 1
lam[, 3] <- 3 * X_train[, 3]^2
lam[, 4] <- 5 * (X_train[, 4] * X_train[, 5])
lam[, 5] <- 2 * (X_train[, 5] + 2 * X_train[, 6])
lam[, 6] <- 2 * (X_train[, 1] + X_train[, 3] - X_train[, 5])
lamt[, 1] <- abs(2 * X_test[, 1] - X_test[, 2])
lamt[, 2] <- 1
lamt[, 3] <- 3 * X_test[, 3]^2
lamt[, 4] <- 5 * (X_test[, 4] * X_test[, 5])
lamt[, 5] <- 2 * (X_test[, 5] + 2 * X_test[, 6])
lamt[, 6] <- 2 * (X_test[, 1] + X_test[, 3] - X_test[, 5])


# vary s to make the problem harder s < 1 or easier s > 2
s <- 1
pr <- exp(s * lam)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_train <- sapply(1:n, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))

pr <- exp(s * lamt)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_test <- sapply(1:nt, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))



# num_sweeps = ceiling(200/log(n))
num_sweeps <- 20
burnin <- 3
num_trees <- 3
max_depth <- 2
mtry <- NULL # round((p + p_cat)/3)
#########################  parallel ####################3
tm <- proc.time()
fit <- XBART.multinomial(
  y = matrix(y_train), num_class = k, X = X_train, Xtest = X_test,
  num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth,
  num_cutpoints = NULL, alpha = 0.95, beta = 1.25, tau_a = 1, tau_b = 1,
  no_split_penality = 1, burnin = burnin, mtry = mtry, p_categorical = p_cat,
  kap = 1, s = 1, verbose = FALSE, set_random_seed = FALSE,
  random_seed = NULL, sample_weights = TRUE, separate_tree = TRUE
)

# the warm start bart only draws 100 samples, without burnin, thinning = 1
fit2 <- mlbart_ini(fit$treedraws, X_train, y_train, k, type = "separate", ntree = num_trees, ndpost = 100, nskip = 0, keepevery = 1)


tm <- proc.time() - tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"], 3), " seconds"), "\n")
# take average of all sweeps, discard burn-in
# a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
a <- apply(fit$yhats_test[burnin:num_sweeps, , ], c(2, 3), median)
pred <- apply(a, 1, which.max) - 1
yhat <- apply(a, 1, which.max) - 1
cat(paste("xbart classification accuracy: ", round(mean(y_test == yhat), 3)), "\n")