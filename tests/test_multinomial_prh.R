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



library(XBART)
library(xgboost)
library(ranger)


seed = 10


#set.seed(seed)


n = 10000
nt = 1000
p = 20
k = 3
lam = matrix(0,n,k)
X_train = matrix(runif(n*p,-1,1), nrow=n)
#logodds = pmax(5*X_train-2.5)
lam[,1] = 2*abs(2*X_train[,1] - X_train[,2])
lam[,2] = 1
lam[,3] = 3*X_train[,3]^2
pr = exp(lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
# logodds = 2*X_train[,1]*X_train[,2]
#pr = plogis(logodds)
#y_train = rbinom(n, 1, pr)
y_train = sapply(1:n,function(j) sample(0:(k-1),1,prob=pr[j,]))

lam = matrix(0,nt,k)

X_test = matrix(runif(nt*p,-1,1), nrow=nt)

#logodds = pmax(5*X_test-2.5)
#logodds = 3*X_test[,3]*(X_test[,1] > X_test[,2]) - 3*(1-X_test[,3])*(X_test[,1] < X_test[,2])

# logodds = 2*X_test[,1]*X_test[,2]
#pr = plogis(logodds)
#y_test = rbinom(nt, 1, pr)
lam[,1] = 2*abs(2*X_test[,1] - X_test[,2])
lam[,2] = 1
lam[,3] = 3*X_test[,3]^2
pr = exp(lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
# logodds = 2*X_train[,1]*X_train[,2]
#pr = plogis(logodds)
#y_train = rbinom(n, 1, pr)
y_test = sapply(1:nt,function(j) sample(0:(k-1),1,prob=pr[j,]))

num_sweeps = 40
burnin = 10


if(0){
# insample error 
y_test = y_train
X_test = X_train
}else{

}
num_trees = 10
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=3, X=X_train, Xtest=X_test, 
            num_trees=num_trees, num_sweeps=num_sweeps, max_depth=250, 
            Nmin=10, num_cutpoints=100, alpha=0.95, beta=2, tau=100/num_trees, tau_later = 100/num_trees,
            no_split_penality = 1, burnin = burnin, mtry = 3, p_categorical = 0L, 
            kap = 1, s = 1, verbose = FALSE, parallel = FALSE, set_random_seed = FALSE, 
            random_seed = NULL, sample_weights_flag = TRUE, draw_tau = FALSE, num_tree_fix = 2, tree_burnin = 1) 

# number of sweeps * number of observations * number of classes
#dim(fit$yhats_test)
tm = proc.time()-tm
cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
# take average of all sweeps, discard burn-in
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
pred = apply(a,1,which.max)-1


# tm2 = proc.time()
# fit2 = XBART.multinomial(y=matrix(y_train), num_class=3, X=X_train, Xtest=X_test, 
#             num_trees=num_trees, num_sweeps=num_sweeps, max_depth=250, 
#             Nmin=10, num_cutpoints=100, alpha=0.95, beta=1.25, tau=50/num_trees, tau_later = 50,
#             no_split_penality = 1, burnin = burnin, mtry = 3, p_categorical = 0L, 
#             kap = 1, s = 1, verbose = FALSE, parallel = FALSE, set_random_seed = FALSE, 
#             random_seed = NULL, sample_weights_flag = TRUE, draw_tau = TRUE, MH_step_size = 0.01, num_tree_fix = 2, tree_burnin = 1) 

# # number of sweeps * number of observations * number of classes
# #dim(fit$yhats_test)
# tm2 = proc.time()-tm2
# cat(paste("\n", "xbart drawing tau runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
# # take average of all sweeps, discard burn-in
# a2 = apply(fit2$yhats_test[burnin:num_sweeps,,], c(2,3), median)
# pred2 = apply(a2,1,which.max)-1

# final predcition
#pred = as.numeric(a[,1] < a[,2])


# Compare with BART probit
#fit2 = pbart(X_train, y_train)

#pred2 = predict(fit2, X_test)
#pred2 = as.numeric(pred2$prob.test.mean > 0.5)



# Compare with ranger
data = data.frame( y = y_train, X = X_train)
data.test = data.frame(X = X_test)
tm = proc.time()
fit3 = ranger(as.factor(y) ~ ., data = data,probability=TRUE, num.trees = 1000)



pred3 = predict(fit3, data.test)$predictions
tm = proc.time()-tm
cat(paste("ranger runtime: ", round(tm["elapsed"],3)," seconds","\n"))


tm = proc.time()
fit.xgb <- xgboost(data = X_train,label=y_train,
                          num_class=3,
                          verbose = 0,
                          max_depth = 4,
                          subsample = 0.80,
                          nrounds=500,
                          early_stopping_rounds = 2,
                          eta = 0.1,
                          params=list(objective="multi:softprob"))

tm = proc.time()-tm
cat(paste("xgboost runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol=k, byrow=TRUE)

yhat.xgb <- max.col(phat.xgb) - 1



#plotROC(pred3$predictions,y_test)
#plotROC(a[,2],y_test,add=TRUE,col='orange')

#pred3 = as.numeric(pred3$predictions > 0.5)



# OUT SAMPLE error
#print(mean(pred == y_test))
#sum(pred2 == y_test)
#print(mean(pred3 == y_test))

cat(paste("xbart rmse on probabilities: ", round(sqrt(mean((a-pr)^2)),3)),"\n")
# cat(paste("xbart sampling tau rmse on probabilities: ", round(sqrt(mean((a2-pr)^2)),3)),"\n")
cat(paste("ranger rmse on probabilities: ", round(sqrt(mean((pred3-pr)^2)),3)),"\n")
cat(paste("xgboost rmse on probabilities: ", round(sqrt(mean((phat.xgb-pr)^2)),3)),"\n")

#par(mfrow=c(1,3))
#plot(pred3[,1],pr[,1],pch=20,cex=0.5)
#plot(pred3[,2],pr[,2],pch=20,cex=0.5)
#plot(pred3[,3],pr[,3],pch=20,cex=0.5)



yhat = apply(a,1,which.max)-1
# yhat2 = apply(a2, 1, which.max)-1
yhat.rf = apply(pred3,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
# cat(paste("xbart sampling tau classification accuracy: ",round(mean(y_test == yhat2),3)),"\n")
cat(paste("ranger classification accuracy: ", round(mean(y_test == yhat.rf),3)),"\n")
cat(paste("xgboost classification accuracy: ", round(mean(yhat.xgb == y_test),3)),"\n")




