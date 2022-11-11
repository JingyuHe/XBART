library(XBART)
library(xgboost)

n.train <- 10000
n.test <- 1000
n.all <- n.train + n.test
p <- 5
p_add <- 10
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

num_sweeps <- 20
burnin <- 3
num_trees <- 15
max_depth <- 25
mtry <- p + p_add


### Compare to not keeping large trees (set tree_size large)
tm2 <- proc.time()
fit2 <- XBART.multinomial(
    y = matrix(y_train), num_class = num_class, X = X_train,
    num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth, update_weight = TRUE,
    num_cutpoints = 40, burnin = burnin, mtry = NULL, p_categorical = p_cat, 
    tau_a = (num_trees * 2 / 2.5^2 + 0.5), tau_b = (num_trees * 2 / 2.5^2), 
    verbose = T, separate_tree = FALSE, update_tau = FALSE, update_phi = FALSE, 
    a = 1 / num_class, no_split_penalty = 0.5, alpha = 0.95, beta = 2, Nmin = 15 * num_class, weight = 2.5, MH_step = 0.05, parallel = F,
    tree_size = 10000
)
tm2 <- proc.time() - tm2

cat(paste("XBART runtime: ", round(tm2["elapsed"], 3), " seconds"), "\n")
pred2 <- predict(fit2, X_test, burnin = burnin)
phat2 <- apply(pred2$yhats[burnin:num_sweeps, , ], c(2, 3), mean)
yhat2 <- pred2$label
spr.xbart2 <- split(phat2, row(phat2))
logloss2 <- sum(mapply(function(x, y) -log(x[y]), spr.xbart2, y_test + 1, SIMPLIFY = TRUE))

cat(paste("XBART accuracy: (not keeping trees)", round(mean(y_test == yhat2), 3)), "\n")


tm <- proc.time()
fit <- XBART.multinomial(
    y = matrix(y_train), num_class = num_class, X = X_train,
    num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth, update_weight = TRUE,
    num_cutpoints = 40, burnin = burnin, mtry = NULL, p_categorical = p_cat, 
    tau_a = (num_trees * 2 / 2.5^2 + 0.5), tau_b = (num_trees * 2 / 2.5^2), 
    verbose = T, separate_tree = FALSE, update_tau = FALSE, update_phi = FALSE, 
    a = 1 / num_class, no_split_penalty = 0.5, alpha = 0.95, beta = 2, Nmin = 15 * num_class, weight = 2.5, MH_step = 0.05, parallel = F,
    tree_size = 50, extra_trees = 100
)
tm <- proc.time() - tm


cat(paste("XBART runtime (keeping trees): ", round(tm["elapsed"], 3), " seconds"), "\n")
cat(paste("XBART runtime (not keeping trees): ", round(tm2["elapsed"], 3), " seconds"), "\n")
pred <- predict(fit, X_test, burnin = burnin)
phat <- apply(pred$yhats[burnin:num_sweeps, , ], c(2, 3), mean)
yhat <- pred$label
spr.xbart <- split(phat, row(phat))
logloss <- sum(mapply(function(x, y) -log(x[y]), spr.xbart, y_test + 1, SIMPLIFY = TRUE))

phat.train <- apply(fit$yhats_train[burnin:num_sweeps,,],c(2,3),mean)
yhat.train <- max.col(phat.train)-1
cat(paste("XBART Insample accuracy (keep large trees) :", round(mean(y_train == yhat.train), 3)), "\n")
cat(paste("XBART Outsample accuracy (keep large trees) : ", round(mean(y_test == yhat), 3)), "\n")
cat(paste("XBART accuracy: (not keeping trees)" , round(mean(y_test == yhat2), 3)), "\n")

print("weight (not keep trees):")
print(fit2$weight[1,])

print("weight (keep trees):")
print(fit$weight[1,])

tm3 <- proc.time()
fit.xgb <- xgboost(
    data = X_train, label = y_train,
    num_class = num_class,
    verbose = 0,
    # max_depth = 15,
    # subsample = 0.80,
    nrounds = 150,
    early_stopping_rounds = 2,
    # eta = 0.1,
    params = list(objective = "multi:softprob")
)
tm3 <- proc.time() - tm3
cat(paste("XGBoost runtime: ", round(tm3["elapsed"], 3), " seconds"), "\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol = num_class, byrow = TRUE)

yhat.xgb <- max.col(phat.xgb) - 1

spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x, y) -log(x[y]), spr, y_test + 1, SIMPLIFY = TRUE))


time <- c(tm["elapsed"], tm3["elapsed"])
lloss <- c(logloss, logloss.xgb)
acc <- c(mean(y_test == yhat), mean(y_test == yhat.xgb))

results = rbind(time, lloss, acc)

rownames(results) = c("Time", "LogLoss", "Accuracy")
colnames(results) <- c("XBART keep large trees", "XGBoost")

print(results)
