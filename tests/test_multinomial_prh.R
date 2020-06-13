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

seed = 10
set.seed(seed)


n = 10000
nt = 5000
p = 20
p_cat = 0
k = 6
lam = matrix(0,n,k)
lamt = matrix(0,nt,k)

X_train = matrix(runif(n*p,-1,1), nrow=n)
X_test = matrix(runif(nt*p,-1,1), nrow=nt)

X_train = cbind(X_train, matrix(rpois(n*p_cat, 20), nrow=n))
X_test = cbind(X_test, matrix(rpois(nt*p_cat, 20), nrow=nt))

p = p+p_cat


lam[,1] = abs(2*X_train[,1] - X_train[,2])
lam[,2] = 1
lam[,3] = 3*X_train[,3]^2
lam[,4] = 5*(X_train[, 4] * X_train[,5])
lam[,5] = 2*(X_train[,5] + 2*X_train[,6])
lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
lamt[,1] = abs(2*X_test[,1] - X_test[,2])
lamt[,2] = 1
lamt[,3] = 3*X_test[,3]^2
lamt[,4] = 5*(X_test[,4]*X_test[,5])
lamt[,5] = 2*(X_test[,5] + 2*X_test[,6])
lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])
pr = exp(10*lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_train = sapply(1:n,function(j) sample(0:(k-1),1,prob=pr[j,]))

pr = exp(10*lamt)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_test = sapply(1:nt,function(j) sample(0:(k-1),1,prob=pr[j,]))



num_sweeps = 20
burnin = 5

if(0){
  # insample error 
  y_test = y_train
  X_test = X_train
}else{
  
}
num_trees = 10

#########################  parallel ####################3
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=250, 
                        Nmin=10, num_cutpoints=100, alpha=0.95, beta=1.25, tau_a = 2.5, tau_b = 0.5,
                        no_split_penality = 1, weight = seq(1, 20, 0.5), burnin = burnin, mtry = 6, p_categorical = p_cat, 
                        kap = 1, s = 1, verbose = TRUE, set_random_seed = FALSE, random_seed = NULL,
                        sample_weights_flag = TRUE, stop_threshold = 0.09, nthread = 0) 


tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
# take average of all sweeps, discard burn-in
# a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
pred = apply(a,1,which.max)-1
yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")


#######################################################


# # Compare with ranger
# # data = data.frame( y = y_train, X = X_train)
# # data.test = data.frame(X = X_test)
# # tm = proc.time()
# # fit3 = ranger(as.factor(y) ~ ., data = data,probability=TRUE, num.trees = 1000)
# # 
# # 
# # 
# # pred3 = predict(fit3, data.test)$predictions
# # tm = proc.time()-tm
# # cat(paste("ranger runtime: ", round(tm["elapsed"],3)," seconds","\n"))
# 
# # 
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


cat(paste("xbart rmse on probabilities: ", round(sqrt(mean((a-pr)^2)),3)),"\n")
# cat(paste("ranger rmse on probabilities: ", round(sqrt(mean((pred3-pr)^2)),3)),"\n")
cat(paste("xgboost rmse on probabilities: ", round(sqrt(mean((phat.xgb-pr)^2)),3)),"\n")

# par(mfrow=c(1,3))
# plot(a[,1],pr[,1],pch=20,cex=0.75)
# plot(a[,2],pr[,2],pch=20,cex=0.75)
# plot(a[,3],pr[,3],pch=20,cex=0.75)

yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("xgboost classification accuracy: ", round(mean(yhat.xgb == y_test),3)),"\n")

spr <- split(a, row(a))
logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))
spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

cat(paste("xbart logloss : ",round(logloss,3)),"\n")
cat(paste("xgboost logloss : ", round(logloss.xgb,3)),"\n")

cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")

table(fit$weight)

cat("early stops per tree: ", round(fit$num_stops/num_sweeps/num_trees, 3), "\n")

fit$importance
# 
# stop_profiler()

