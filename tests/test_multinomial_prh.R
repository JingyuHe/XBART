# install.packages("D:/XBART",repos=NULL,type="source")
library(XBART)
library(xgboost)
# library(ranger)

# start_profiler('test_multinomial_prh.log')
# 

plotROC <- function(pihat,ytrue,add=FALSE,col = 'steelblue'){
  
  thresh <- sort(pihat)
  N <- length(pihat)
  yhat <- sapply(1:N, function(a) as.double(pihat >= thresh[a]))
  tpr <- sapply(1:N, function(a) length(which(ytrue==1 & yhat[,a]==1))/sum(ytrue==1))
  fpr <- sapply(1:N, function(a) length(which(ytrue==0 & yhat[,a]==1))/sum(ytrue==0));
  if (add == FALSE){
    plot(fpr,tpr,pch=20,cex=0.8,col=col,bty='n',type='b')
    abline(a=0,b=1,lty=2)
  }else{
    points(fpr,tpr,pch=20,cex=0.8,col=col,bty='n',type='b')
  }
  #print(mean(tpr))
}

get_entropy <- function(nclass){
  pi = c(0.99, rep(0.01/(nclass-1), nclass-1))
  pi = pi/sum(pi)
  return (sum(-pi*log(pi)))
}

#seed = 10
#set.seed(seed)

# 
# n = 200
# nt = 50
n = 5000
nt = 1000
p = 6
p_cat = 0
k = 6
lam = matrix(0,n,k)
lamt = matrix(0,nt,k)


K = matrix(rnorm(3*p),p,3)
X_train = t(K%*%matrix(rnorm(3*n),3,n))
X_test = t(K%*%matrix(rnorm(3*nt),3,nt))

X_train = pnorm(X_train)
X_test = pnorm(X_test)

#X_train = matrix(runif(n*p,-1,1), nrow=n)
#X_test = matrix(runif(nt*p,-1,1), nrow=nt)

X_train = cbind(X_train, matrix(rbinom(n*p_cat, 1, 0.5), nrow = n))
X_test = cbind(X_test, matrix(rbinom(nt*p_cat, 1, 0.5), nrow = nt))

# X_train = cbind(X_train, matrix(rpois(n*p_cat, 20), nrow=n))
# X_test = cbind(X_test, matrix(rpois(nt*p_cat, 20), nrow=nt))


lam[,1] = abs(3*X_train[,1] - X_train[,2])
lam[,2] = 2
lam[,3] = 3*X_train[,3]^2
lam[,4] = 4*(X_train[, 4] * X_train[,5])
lam[,5] = 2*(X_train[,5] + X_train[,6])
lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
lamt[,1] = abs(3*X_test[,1] - X_test[,2])
lamt[,2] = 2
lamt[,3] = 3*X_test[,3]^2
lamt[,4] = 4*(X_test[,4]*X_test[,5])
lamt[,5] = 2*(X_test[,5] + X_test[,6])
lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])

# lam[,1] = 3*abs(2*X_train[,1] - X_train[,2])
# lam[,2] = 2
# lam[,3] = 3*X_train[,3]^2
# lam[,4] = 5*(X_train[, 4] * X_train[,5])
# lam[,5] = (X_train[,5] + 2*X_train[,6])
# lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
# lamt[,1] = 3*abs(2*X_test[,1] - X_test[,2])
# lamt[,2] = 2
# lamt[,3] = 3*X_test[,3]^2
# lamt[,4] = 5*(X_test[,4]*X_test[,5])
# lamt[,5] = (X_test[,5] + 2*X_test[,6])
# lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])


# vary s to make the problem harder s < 1 or easier s > 2
s = 10
pr = exp(s*lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_train = sapply(1:n,function(j) sample(0:(k-1),1,prob=pr[j,]))

pr = exp(s*lamt)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_test = sapply(1:nt,function(j) sample(0:(k-1),1,prob=pr[j,]))



# num_sweeps = ceiling(200/log(n)) 
num_sweeps = 20
burnin = 5
num_trees = 20
#########################  parallel ####################3
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, p_categorical = p_cat, 
                        separate_tree = FALSE)


tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
# take average of all sweeps, discard burn-in
# a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
# a = a / rowSums(a) # bc a is the median of pi, need normalization
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
pred = apply(a,1,which.max)-1
yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")


# you may use the predict function as well
fitted = predict.XBARTmultinomial(fit, X_test)
a = apply(fitted$yhats[burnin:num_sweeps,,], c(2,3), mean)
pred = apply(a,1,which.max)-1
yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")



tm2 = proc.time()
fit.xgb <- xgboost(data = X_train, label=y_train,
                   num_class=k,
                   verbose = 0,
                   max_depth = 4,
                   subsample = 0.80,
                   nrounds=500,
                   early_stopping_rounds = 2,
                   eta = 0.1,
                   params=list(objective="multi:softprob"))

tm2 = proc.time()-tm2
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol=k, byrow=TRUE)

yhat.xgb <- max.col(phat.xgb) - 1


spr <- split(a, row(a))
logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))
spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

cat(paste("xbart logloss : ",round(logloss,3)),"\n")
cat(paste("xgboost logloss : ", round(logloss.xgb,3)),"\n")

cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")

yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("xgboost classification accuracy: ", round(mean(yhat.xgb == y_test),3)),"\n")


cat("importance ", fit$importance, "\n")

# par(mfrow = c(1, 2))
plot(as.vector(fit$weight))
# plot(as.vector(fit$tau_a))
summary(as.vector(fit$weight))
# stop_profiler()
