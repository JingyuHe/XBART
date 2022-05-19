library(XBART)
library(xgboost)
library(ranger)
# path = '~/Dropbox/MNIST/'
path <- "~/mnist/"
load("~/mnist/features.rda")
nthread <- 12
# D <- read.csv(paste(path,'mnist_train.csv', sep=''),header=FALSE)
# y = D[,1]
# D = D[,-1]
# #
# Dtest <- read.csv(paste(path, 'mnist_test.csv', sep =''),header=FALSE)
# ytest = Dtest[,1]
# Dtest = Dtest[,-1]
# pred = matrix(0,10000,10)
#
# X_train = D
# X_test = Dtest
p <- ncol(X_train)


X.train <- X_train
X.test <- X_test

v <- 0
for (h in 1:p) {
  breaks <- unique(as.numeric(quantile(c(X_train[, h], X_test[, h]), seq(0, 1, length.out = 4))))
  # breaks = seq(min(c(X_train[,h],X_test[,h])),max(c(X_train[,h],X_test[,h])),length.out = 25)
  breaks <- c(-Inf, breaks, +Inf)
  # print(breaks)
  if (length(breaks) > 3) {
    v <- v + 1
    X.train[, v] <- cut(X_train[, h], breaks = breaks, include.lowest = TRUE, labels = FALSE)
    X.test[, v] <- cut(X_test[, h], breaks = breaks, include.lowest = TRUE, labels = FALSE)
  }
}

# print(v)
X_train <- X.train[, 1:v]
X_test <- X.test[, 1:v]
p <- v



num_sweeps <- 20
num_trees <- 50
burnin <- 3
max_depth <- 10
mtry <- ceiling(p / 2)

###################### xbart  #################
t <- proc.time()
fit <- XBART.multinomial(
  y = matrix(y), num_class = 10, X = X_train, Xtest = X_test,
  num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth,
  burnin = burnin, mtry = mtry, p_categorical = p,
  verbose = TRUE, parallel = TRUE, nthread = nthread,
  separate_tree = FALSE
)
t <- proc.time() - t

pred <- apply(fit$yhats_test[(burnin):(num_sweeps - 0), , ], c(2, 3), mean)
yhat <- max.col(pred) - 1

spr <- split(pred, row(pred))
logloss <- sum(unlist(mapply(function(x, y) -log(x[y]), spr, ytest, SIMPLIFY = TRUE)))

cat("running time ", t[3], " seconds \n")

cat("XBART error rate ", mean(yhat != ytest), "\n")

cat(paste("xbart logloss : ", round(logloss, 3)), "\n")

##################################################################


if (1) {
  tt <- list()
  acc <- list()
  logloss <- list()
  tt$xbart <- t
  acc$xbart <- mean(yhat != ytest)
  logloss$xbart <- logloss

  t_xgb <- proc.time()
  fit.xgb <- xgboost(
    data = as.matrix(X_train), label = matrix(y),
    num_class = 10, verbose = 0, nthread = nthread,
    nrounds = 500,
    params = list(objective = "multi:softprob")
  )
  phat.xgb <- predict(fit.xgb, as.matrix(X_test))
  t_xgb <- proc.time() - t_xgb
  tt$xgb <- t_xgb[3]

  phat.xgb <- matrix(phat.xgb, ncol = 10, byrow = TRUE)
  yhat.xgb <- max.col(phat.xgb) - 1
  acc$xgb <- mean(yhat.xgb != ytest)
  spr.xgb <- split(phat.xgb, row(phat.xgb))
  logloss$xgb <- sum(mapply(function(x, y) -log(x[y]), spr.xgb, ytest + 1, SIMPLIFY = TRUE))

  t_ranger <- proc.time()
  fit.ranger <- ranger(as.factor(y) ~ ., data = data.frame(y = y, X = X_train), probability = TRUE, num.threads = nthread)
  pred.ranger <- predict(fit.ranger, data.frame(X = X_test))$predictions
  t_ranger <- proc.time() - t_xgb
  tt$ranger <- t_ranger[3]

  yhat.ranger <- max.col(pred.ranger) - 1
  acc$ranger <- mean(yhat.ranger != ytest)
  spr.ranger <- split(pred.ranger, row(pred.ranger))
  logloss$ranger <- sum(mapply(function(x, y) -log(x[y]), spr.ranger, ytest + 1, SIMPLIFY = TRUE))
}

remove(X_train, X_test, X.train, X.test)
save.image(paste(path, "mnist_result/123001.rda", sep = ""))