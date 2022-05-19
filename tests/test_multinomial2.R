# plotROC <- function(pihat,ytrue,add=FALSE,col = 'steelblue'){
  
#   thresh <- sort(pihat)
#   N <- length(pihat)
#   yhat <- sapply(1:N, function(a) as.double(pihat >= thresh[a]))
#   tpr <- sapply(1:N, function(a) length(which(ytrue==1 & yhat[,a]==1))/sum(ytrue==1))
#   fpr <- sapply(1:N, function(a) length(which(ytrue==0 & yhat[,a]==1))/sum(ytrue==0));
#   if (add == FALSE){
#     plot(fpr,tpr,pch=20,cex=0.8,col=col,bty='n',type='b')
#     abline(a=0,b=1,lty=2)
#   }else{
#     points(fpr,tpr,pch=20,cex=0.8,col=col,bty='n',type='b')
#   }
#   #print(mean(tpr))
# }
# ​
​
​
library(XBART)
library(BART)
library(ranger)
​
​
seed = 10
​
​
set.seed(seed)
​
​
n = 1000
nt = 1000
p = 5
k = 3
lam = matrix(0,n,k)
X_train = matrix(runif(n*p), nrow=n)
#logodds = pmax(5*X_train-2.5)
lam[,1] = 2*X_train[,3]*(X_train[,1] > X_train[,2]) 
lam[,2] = -(1-X_train[,3])*(X_train[,1] < X_train[,2])
lam[,3] = X_train[,3] + X_train[,1]^2
pr = exp(lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
# logodds = 2*X_train[,1]*X_train[,2]
#pr = plogis(logodds)
#y_train = rbinom(n, 1, pr)
y_train = sapply(1:n,function(j) sample(1:3,1,prob=pr[j,])) - 1
​
lam = matrix(0,nt,k)
​
X_test = matrix(runif(nt*p), nrow=nt)
​
#logodds = pmax(5*X_test-2.5)
#logodds = 3*X_test[,3]*(X_test[,1] > X_test[,2]) - 3*(1-X_test[,3])*(X_test[,1] < X_test[,2])
​
# logodds = 2*X_test[,1]*X_test[,2]
#pr = plogis(logodds)
#y_test = rbinom(nt, 1, pr)
lam[,1] = 2*X_test[,3]*(X_test[,1] > X_test[,2]) 
lam[,2] = -(1-X_test[,3])*(X_test[,1] < X_test[,2])
lam[,3] = X_test[,3] + X_test[,1]^2
pr = exp(lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
# logodds = 2*X_train[,1]*X_train[,2]
#pr = plogis(logodds)
#y_train = rbinom(n, 1, pr)
y_test = sapply(1:nt,function(j) sample(1:3,1,prob=pr[j,])) - 1
​
num_sweeps = 30
burnin = 15
​
​
if(0){
# insample error 
y_test = y_train
X_test = X_train
}else{
​
}
num_trees = 30
# tm = proc.time()
fit = XBART.multinomial(y=y_train, num_class=3, X=X_train, Xtest=X_test, 
            num_trees=num_trees, num_sweeps=num_sweeps, max_depth=250, 
            Nmin=1, num_cutpoints=100, alpha=0.95, beta=2.0, tau=8/num_trees, 
            no_split_penality = "Auto", burnin = 1L, mtry = p, p_categorical = 0L, 
            kap = 1, s = 1, verbose = TRUE, parallel = FALSE, set_random_seed = FALSE, 
            random_seed = seed, sample_weights = TRUE) 
​
# number of sweeps * number of observations * number of classes
# dim(fit$yhats_test)
# tm = proc.time()-tm
# print(tm)
​
# take average of all sweeps, discard burn-in
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
​
​
# final predcition
pred = apply(a, 1, which.max) - 1
​
# Compare with ranger
data = data.frame( y = y_train, X = X_train)
data.test = data.frame(X = X_test)
​
fit3 = ranger(as.factor(y) ~ ., data = data)
​
pred3 = predict(fit3, data.test)
​



​
plotROC(pred3$predictions,y_test)
plotROC(a[,2],y_test,add=TRUE,col='orange')
​
pred3 = as.numeric(pred3$predictions) - 1
​
​
​
# OUT SAMPLE error
sum(pred == y_test)
#sum(pred2 == y_test)
sum(pred3 == y_test)