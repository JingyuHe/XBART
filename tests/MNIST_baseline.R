# library(XBART)
library(xgboost)

D <- read.csv('~/mnist/mnist_train.csv',header=FALSE)
y = D[,1]
D = D[,-1]
#
Dtest <- read.csv('~/mnist/mnist_test.csv',header=FALSE)
ytest = Dtest[,1]
Dtest = Dtest[,-1]
#
#
k = 3
X = matrix(NA,28*28,k*10)
for (h in 0:9){
  print(h)
  S = svd(t(D[y==h,]))
  X[,(h*k):(h*k+k-1)+1] = S$u[,1:k]
}


XXinv = solve(t(X)%*%X)
P = XXinv%*%t(X)
X_train = t(P%*%t(D))
X_test = t(P%*%t(Dtest))
p = ncol(X_train)

#load("mnist_data.rda")

# X_train = X_train + 0.0001*rnorm(ncol(X_train)*nrow(X_train))
# X_test = X_test + 0.0001*rnorm(ncol(X_test)*nrow(X_test))

# for (h in 1:p){
# breaks =unique(as.numeric(quantile(c(X_train[,h],X_test[,h]),seq(0,1,length.out=20))))
# breaks = seq(min(c(X_train[,h],X_test[,h])),max(c(X_train[,h],X_test[,h])),length.out = 25)
# 
# print(breaks)
# X_train[,h] = cut(X_train[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
# X_test[,h] = cut(X_test[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
# }
# X_train = X_train[1:1000,]
# y = y[1:1000]


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