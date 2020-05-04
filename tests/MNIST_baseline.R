library(XBART)
library(xgboost)

D <- read.csv('~/mnist/mnist_train.csv',header=FALSE)
y = D[,1]
D = D[,-1]
# 
Dtest <- read.csv('~/mnist/mnist_test.csv',header=FALSE)
ytest = Dtest[,1]
Dtest = Dtest[,-1]
pred = matrix(0,10000,10)

X_train = D
X_test = Dtest
p = ncol(X_train)

# for (i in 1:p){
#   X_train[, i] = X_train[, i] + 0.01*rnorm(length(y))
#   X_test[, i] = X_test[, i] + 0.01*rnorm(length(ytest))
# }


X.train = X_train
X.test = X_test

v = 0
for (h in 1:p){
  breaks =unique(as.numeric(quantile(c(X_train[,h],X_test[,h]),seq(0,1,length.out=4))))
  #breaks = seq(min(c(X_train[,h],X_test[,h])),max(c(X_train[,h],X_test[,h])),length.out = 25)
  breaks = c(-Inf,breaks,+Inf)
  #print(breaks)
  if (length(breaks)>3){
    v = v + 1
    X.train[,v] = cut(X_train[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
    X.test[,v] = cut(X_test[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
  }
}

#print(v)
X_train = X.train[,1:v]
X_test = X.test[,1:v]
p = v

X_train[,1] = X_train[,1] + 0.01*rnorm(length(y))

X_train = as.matrix(X_train)
X_test = as.matrix(X_test)

t = proc.time()

xgb.basic.mod1 <- xgboost(data = X_train,label=y,
                          num_class=10,
                          max_depth = 5,
                          subsample = 0.80,
                          nrounds=50,
                          early_stopping_rounds = 2,
                          eta = 0.9,
                          params=list(objective="multi:softprob"))

xgb.basic.pred <- predict(xgb.basic.mod1, X_test)
xgb.basic.pred <- matrix(xgb.basic.pred, ncol=10, byrow=TRUE)
pred.xgb <- max.col(xgb.basic.pred) - 1

spr <- split(xgb.basic.pred, row(xgb.basic.pred))
logloss.xgb <- sum(unlist(mapply(function(x,y) -log(x[y]), spr, ytest, SIMPLIFY =TRUE)))

cat("running time ", t[3], " seconds \n")

cat("xgboost error rate ", mean(pred.xgb != ytest), "\n")

cat(paste("xgboost logloss : ", round(logloss.xgb,3)),"\n")


for(i in 0:9){
  cat("xgboost error rate in ", i, ": ", round(mean(pred.xgb[ytest==i]!=i), 4), 
      " misclassified as ", tail(names(sort(table(pred.xgb[ytest==i]))), 2)[1], "\n " )
}