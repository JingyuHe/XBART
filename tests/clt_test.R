library("dbarts")
library("XBART")
library("xgboost")
library("pROC")

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
  print(mean(tpr))
}




set.seed(2003)
n = 5000
nt = 1000
x1 = runif(n,-2,2)
x2 = runif(n,-2,2)
dcat = 0
 
xtest1 = runif(nt,-2,2)
xtest2 = runif(nt,-2,2)

f = function(x1,x2){
  
 # y = 2*abs(x1)/(abs(x2)+1) >= abs(x2)/(abs(x1)+1)
  
  y = x1^2 + x2^2 > 0.5 & x1^2 + x2^2 < 2
  
  
#  x1^2 + x2 - 1 >= x2^2 - 2*x1 
  
}

#r = (1-rbinom(length(x1),1,0.75*(1-pnorm(abs(x2)))))
r = rbinom(length(x1),1,0.9*f(x1,x2) + 0.7*(1-f(x1,x2)))
#r = 1
y = f(x1,x2)*r + (1-r)*(1-f(x1,x2))

ytest = f(xtest1,xtest2)

plot(x1,x2,pch=20,col=y+1)


x = cbind(x1,x2,matrix(runif(n*10,-2,2),n,10))
xtest = cbind(xtest1,xtest2,matrix(runif(nt*10,-2,2),nt,10))


### BART ###
t = proc.time()
fit = bart(x,as.numeric(y),xtest,ndpost=2000,nskip = 1000)
phat = colMeans(pnorm(fit$yhat.test))
t = proc.time() - t
print(t)
mean((ytest == (phat>0.5)))

### XBART ###
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 10,
    L = 1,
    nsweeps = 50,
    Nmin = 10,
    alpha = 0.95,
    beta = 1.25,
    mtry = 5,
    burnin = 15)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 100;XBART_params$tau = 1/(num_tress)
    XBART_params$a = 0.000001; XBART_params$b = 0.000001;
    return(XBART_params)
}
params = get_XBART_params(n,2,y)
dcat = 0
parl = F

t = proc.time()
out = XBART(2*as.matrix(y)-1 + 0.1*rnorm(n), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = params$Nmin, num_cutpoints = params$Ncutpoints, 
            alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
            mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
            parallel = parl)
 pred = predict(out,as.matrix(xtest))
 t = proc.time() - t
 print(t)
 yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean)


### CLT ### 
t = proc.time()
out = XBART.CLT(2*as.matrix(y)-1 + 0.1*rnorm(n), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = params$Nmin, num_cutpoints = params$Ncutpoints, 
            alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
            mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
            parallel = parl)
pred.probit = predict(out,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.probit.pred = apply(pred.probit[,params$burnin:params$nsweeps],1,mean)


### XGBoost ###
xgboost_parms = list(booster = 'gbtree',objective = 'binary:logistic',silent=T,
  eta=.1,min_child_weight = 10,colsample = 0.7)
t5 = proc.time()
dtrain = xgb.DMatrix(x,label=y)
bst = xgb.train(dtrain,params=xgboost_parms, nrounds = 200) # Fit XGBoost
yhat.train.xgb = predict(bst,newdata=x) 
yhat.test.xgb = predict(bst,newdata=xtest) # Predict on test
t5 = proc.time() - t5


print("BART: ")
print(auc(ytest,phat))
print("XBART: ")
print(auc(ytest,yhat.pred))
print("XBART CLT: ")
print(auc(ytest,yhat.probit.pred))
print("XGBoost: ")
print(auc(ytest,yhat.test.xgb))
plot(phat,yhat.pred)
print("Done!")

plotROC(phat,ytest)
plotROC(yhat.pred,ytest,add=TRUE,col='orange')
plotROC(yhat.probit.pred ,ytest,add=TRUE,col='green')
plotROC(yhat.test.xgb,ytest,add=TRUE,col="purple")


# phat = colMeans(pnorm(fit$yhat.train))
# pred = predict(out,as.matrix(x))
# yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean)
# print(mean((y == (phat>0.5))))
# print(mean((y == (yhat.pred>0.5))))
# plot(phat,yhat.pred)
# print("Done!")
# 
# plotROC(phat,y)
# plotROC(yhat.pred,y,add=TRUE,col='orange')
