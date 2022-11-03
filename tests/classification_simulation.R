library(XBART)
library(xgboost)

n.train <- 5000
n.test <- 1000
n.all <- n.train + n.test
k <- 3
p <- k
p_add <- 0
p_cat <- 0
acc_level <- 0.99
x <- matrix(rnorm(n.all * (p + p_add)), n.all, (p + p_add))


QR <- qr(matrix(rnorm(p * p), p, p))

lam <- diag(qr.R(QR))
Lam <- diag(lam / abs(lam))
R <- qr.Q(QR) %*% Lam

entropy <- function(x) {
    1 / sum(exp(-x * ((0:(p - 1)) / (p - 1))^2)) - acc_level
}
kappa <- uniroot(entropy, c(0, 1000), tol = 1e-5)$root

xprime <- x[, 1:p] %*% R

D <- as.matrix(dist(1:p), p, p) / (p - 1)

error.mat <- exp(-kappa * D^2)


error.mat <- error.mat %*% diag(1 / colSums(error.mat))
# barplot(error.mat[,3])
ind <- apply(xprime, 1, which.max)

y.all <- sapply(1:n.all, function(j) sample(1:p, 1, prob = error.mat[, ind[j]]))

X_train <- x[1:n.train, ]
X_test <- x[-(1:n.train), ]

y_train <- y.all[1:n.train] - 1
y_test <- y.all[-(1:n.train)] - 1

num_class <- max(y_train) + 1

num_sweeps <- 25
burnin <- 10
num_trees <- 50
max_depth <- 10
mtry <- p
tm <- proc.time()
fit <- XBART.multinomial(
    y = matrix(y_train), num_class = k, X = X_train,
    num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth,
    num_cutpoints = NULL, burnin = burnin, mtry = mtry, p_categorical = p_cat, tau_a = (num_trees * 2 / 3.5^2 + 0.5), tau_b = (num_trees * 2 / 3.5^2), verbose = FALSE, separate_tree = FALSE, updte_tau = FALSE, update_weight = TRUE, update_phi = FALSE, a = 2 / k, weight_exponent = 6, no_split_penalty = 0.5, beta = 2, weight = 10, MH_step = 0.1, parallel = FALSE
)
tm <- proc.time() - tm
cat(paste("XBART runtime (sampling weights): ", round(tm["elapsed"], 3), " seconds"), "\n")
pred <- predict(fit, X_test, burnin = burnin)
phat <- apply(pred$yhats[burnin:num_sweeps, , ], c(2, 3), mean)
yhat <- pred$label
spr.xbart <- split(phat, row(phat))
logloss <- sum(mapply(function(x, y) -log(x[y]), spr.xbart, y_test + 1, SIMPLIFY = TRUE))


par(mfrow = c(1, 2))
hist(fit$weight)
plot(c(fit$weight))



tm2 <- proc.time()
fit2 <- XBART.multinomial(
    y = matrix(y_train), num_class = k, X = X_train,
    num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth,
    num_cutpoints = NULL, burnin = burnin, mtry = mtry, p_categorical = p_cat, tau_a = (num_trees * 2 / 3.5^2 + 0.5), tau_b = (num_trees * 2 / 3.5^2), verbose = FALSE, separate_tree = FALSE, updte_tau = FALSE, update_weight = FALSE, update_phi = FALSE, a = 2 / k, weight_exponent = 6, no_split_penalty = 0.5, beta = 2, weight = 2, MH_step = 0.25, parallel = FALSE
)
tm2 <- proc.time() - tm2
cat(paste("XBART runtime: ", round(tm2["elapsed"], 3), " seconds"), "\n")
pred2 <- predict(fit2, X_test, burnin = burnin)
phat2 <- apply(pred2$yhats[burnin:num_sweeps, , ], c(2, 3), mean)
yhat2 <- pred2$label
spr.xbart2 <- split(phat2, row(phat))
logloss2 <- sum(mapply(function(x, y) -log(x[y]), spr.xbart2, y_test + 1, SIMPLIFY = TRUE))


tm3 <- proc.time()
fit.xgb <- xgboost(
    data = X_train, label = y_train,
    num_class = k,
    verbose = 0,
    max_depth = 4,
    subsample = 0.80,
    nrounds = 500,
    early_stopping_rounds = 2,
    eta = 0.1,
    params = list(objective = "multi:softprob")
)
tm3 <- proc.time() - tm3
cat(paste("XGBoost runtime: ", round(tm3["elapsed"], 3), " seconds"), "\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol = k, byrow = TRUE)

yhat.xgb <- max.col(phat.xgb) - 1

spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x, y) -log(x[y]), spr, y_test + 1, SIMPLIFY = TRUE))


time <- c(tm["elapsed"], tm2["elapsed"], tm3["elapsed"])
lloss <- c(logloss, logloss2, logloss.xgb)
acc <- c(mean(y_test == yhat), mean(y_test == yhat2), mean(y_test == yhat.xgb))

results = rbind(time, lloss, acc)

rownames(results) = c("Time", "LogLoss", "Accuracy")
colnames(results) = c("XBART sampling weights", "XBART", "XGBoost")

print(results)



# require(GIGrvg)

# num = 10000000

# n = 50000
# sy = 50000
# c = 8
# d = 8

# eta <- -c + n
# chi <- 2 * d
# psi <- 2 * sy
# t = proc.time()
# a = XBART::test(eta, chi, psi, num)
# t = proc.time() - t
# t
