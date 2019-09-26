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
​
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
#set.seed(seed)
​
​
n = 2000
nt = 2000
p = 3
X_train = matrix(runif(n*p), nrow=n)
logodds = pmax(5*X_train-2.5)
# logodds = X[,1] + X[,2]
pr = plogis(logodds)
y_train = rbinom(n, 1, pr)
​
X_test = matrix(runif(nt*p), nrow=nt)
logodds = pmax(5*X_test-2.5)
# logodds = X[,1] + X[,2]
pr = plogis(logodds)
y_test = rbinom(nt, 1, pr)
​
​
num_sweeps = 40
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
num_trees = 100
tm = proc.time()
fit = XBART.multinomial(y=y_train, num_class=2, X=X_train, Xtest=X_test, 
            num_trees=num_trees, num_sweeps=num_sweeps, max_depth=50, 
            Nmin=5, num_cutpoints=50, alpha=0.95, beta=1.25, tau=100, 
            no_split_penality = log(50), burnin = 1L, mtry = 0L, p_categorical = 0L, 
            kap = 1, s = 1, verbose = FALSE, parallel = FALSE, set_random_seed = FALSE, 
            random_seed = seed, sample_weights_flag = TRUE) 
​
# number of sweeps * number of observations * number of classes
dim(fit$yhats_test)
tm = proc.time()-tm
print(tm)
​
# take average of all sweeps, discard burn-in
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
​
​
# final predcition
pred = as.numeric(a[,1] < a[,2])
​
​
# Compare with BART probit
#fit2 = pbart(X_train, y_train)
​
#pred2 = predict(fit2, X_test)
#pred2 = as.numeric(pred2$prob.test.mean > 0.5)
​
​
​
# Compare with ranger
data = data.frame( y = y_train, X = X_train)
data.test = data.frame(X = X_test)
​
fit3 = ranger(y ~ ., data = data)
​
pred3 = predict(fit3, data.test)
​
​
​
plotROC(pred3$predictions,y_test)
plotROC(a[,2],y_test,add=TRUE,col='orange')
​
pred3 = as.numeric(pred3$predictions > 0.5)
​
​
​
# OUT SAMPLE error
sum(pred == y_test)
#sum(pred2 == y_test)
sum(pred3 == y_test)













































# library(XBART)
# library(BART)
# library(ranger)


# seed = 10


# set.seed(seed)


# n = 7000
# n_train = 5000

# p = 2
# X = matrix(runif(n*p), nrow=n)
# logodds = -1 + 2*X[,1] + X[,2]^2 #- 0.5*X[,1]*X[,2]
# # logodds = X[,1] + X[,2]
# pr = plogis(logodds)
# y = rbinom(n, 1, pr)

# y_train = y[1:n_train]
# y_test = y[(n_train + 1):n]
# X_train = X[1:n_train, ]
# X_test = X[(n_train + 1):n, ]

# num_sweeps = 30
# burnin = 20


# if(0){
# # insample error 
# y_test = y_train
# X_test = X_train
# }else{

# }

# t = proc.time()
# fit = XBART.multinomial(y=y_train, num_class=2, X=X_train, Xtest=X_test, 
#             num_trees=100, num_sweeps=num_sweeps, max_depth=300, 
#             Nmin=5, num_cutpoints=50, alpha=0.95, beta=1.25, tau=1, 
#             no_split_penality = "Auto", burnin = 1L, mtry = 0L, p_categorical = 0L, 
#             kap = 16, s = 4, verbose = FALSE, parallel = FALSE, set_random_seed = TRUE, 
#             random_seed = seed, sample_weights_flag = TRUE) 
# t = proc.time() - t
# # number of sweeps * number of observations * number of classes
# dim(fit$yhats_test)


# # take average of all sweeps, discard burn-in
# a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)


# # final predcition
# pred = as.numeric(a[,1] < a[,2])


# # Compare with BART probit
# fit2 = pbart(X_train, y_train)

# pred2 = predict(fit2, X_test)
# pred2 = as.numeric(pred2$prob.test.mean > 0.5)



# # Compare with ranger
# data = data.frame( y = y_train, X = X_train)
# data.test = data.frame(X = X_test)

# fit3 = ranger(y ~ ., data = data)

# pred3 = predict(fit3, data.test)

# pred3 = as.numeric(pred3$predictions > 0.5)



# # OUT SAMPLE accuracy
# sum(pred == y_test)
# sum(pred2 == y_test)
# sum(pred3 == y_test)
