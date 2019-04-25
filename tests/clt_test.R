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
    #print(mean(tpr))
}




#set.seed(7215)
n = 10000
nt = 2500
#nt = 1000
a = 5

dcat = 0


d = 30
k = d
x = matrix(runif(n*d,-a,a),n,d)
xtest = matrix(runif(nt*d,-a,a),nt,d)

#xtest = x

f = function(xinput){
  


 #rad = sqrt(rowSums(xinput[,1:10]^2))
 #prob = dnorm(rad,6,1)
 #prob = prob/(dnorm(0,0,1)+0.2) +0.01

 # prob[xinput[,30]>0] = 0.5*prob[xinput[,30]>0] + 0.5*0.5
   rad = sqrt(rowSums(xinput[,1:2]^2))
#   prob1 = dnorm(rad,2.5,0.75,log = F)
#   prob1 = prob1/(dnorm(0,0,0.75,log=F)+0.01)  

#  prob = pnorm(sin(rad))*(rad < 10) + 0.5*(rad >10)
   
 prob = pnorm(sin(rad))
   
#prob = 0.2*(rad<2) + 0.9*(rad >2 & rad < 3) + 0.6*(rad>3 & rad < 4) + 0.1*(rad>4) 


#prob = 0.25*prob1 + 0.5*prob + 0.25*0.5
 
   alpha = 1
   prob = (xinput[,3]>1)*prob^alpha + (xinput[,3]<1)*(1-prob)^(alpha)
   prob = prob
 # prob = (xinput[,20]>1)*prob^alpha + (xinput[,20]<1)*(1-prob)^(alpha)
   
  
  
  # prob = 1-prob
   
#   prob[xinput[,30]>0] = 0.2*ind[xinput[,30]>0] + 0.4
# prob = pnorm(rad,6,1)
 
   return(prob)
       

   
   
  #  x1^2 + x2 - 1 >= x2^2 - 2*x1
    
}

#r = (1-rbinom(length(x1),1,0.75*(1-pnorm(abs(x2)))))

#r = rbinom(length(x1),1,0.9*f(x1,x2) + 0.7*(1-f(x1,x2)))
#r = 1
#y = f(x1,x2)*r + (1-r)*(1-f(x1,x2))
k = 0.25
p = f(x)*k+ (1-k)*runif(nrow(x))
p = (p - min(p))/(max(p)-min(p))
y = rbinom(nrow(x),1,p)


ytest = f(xtest) > 1/2

plot(x[,1],x[,2],pch=20,col=y+1)

#y = f(x) + sqrt(f(x)*(1-f(x)))*rnorm(n)




### XBART ###
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 15,
    L = 1,
    nsweeps = 150,
    Nmin = 100,
    alpha = 0.95,
    beta = 1.25,
    mtry = 10,
    burnin = 15)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 100;XBART_params$tau = var(y)/(num_tress)
    XBART_params$a = 0.000001; XBART_params$b = 0.000001;
    return(XBART_params)
}
params = get_XBART_params(n,d,y)
dcat = 0
parl = T

print(parl)


# ### BART ###
#  t = proc.time()
#  fit = bart(x,as.numeric(y),xtest,ndpost=5000,nskip = 5000,ntree = params$M, numcut = params$Ncutpoints, verbose = FALSE)
#  phat = colMeans(pnorm(fit$yhat.test))
#  t = proc.time() - t
#  print(t)


t = proc.time()
out = XBART(as.matrix(y)-mean(y), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = params$Nmin, num_cutpoints = params$Ncutpoints,
alpha = params$alpha, beta = params$beta, tau = var(y)/params$M, s= 1,kap = 1,
mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
parallel = parl,random_seed = 10, no_split_penality = 0)
pred = predict(out,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean) + mean(y)


### CLT ###
get_XBART_params_clt <- function(n,d,y){
   XBART_params = list(M = 15,
   L = 1,
   nsweeps = 150,
   Nmin = 100,
   alpha = 0.95,
   beta = 1.25,
   mtry = 10,
   burnin = 15)
   num_tress = XBART_params$M
   XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
   XBART_params$Ncutpoints = 100;XBART_params$tau = var(y)/(num_tress)
   return(XBART_params)
}
params_clt = get_XBART_params(n,d,y)
t = proc.time()
out_clt = XBART.CLT(matrix(y)-mean(y), as.matrix(x), as.matrix(xtest), num_trees = params_clt$M, L = 1, num_sweeps = params_clt$nsweeps, max_depth = params_clt$max_depth, Nmin = params_clt$Nmin, num_cutpoints = params_clt$Ncutpoints,
alpha = params_clt$alpha, beta = params_clt$beta, tau = params_clt$tau, s= 1,kap = 1,
mtry = params_clt$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
parallel = parl,random_seed = 10)
pred.clt = predict(out_clt,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.clt.pred = apply(pred.clt[,params$burnin:params$nsweeps],1,mean)+mean(y)

# ### Probit ###
# t = proc.time()
# out_probit = XBART.Probit(as.matrix(y), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = 10, num_cutpoints = params$Ncutpoints,
# alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1,
# mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
# parallel = parl)
# pred.probit = predict(out_probit,as.matrix(xtest))
# t = proc.time() - t
# print(t)
# yhat.probit.pred = apply(pnorm(pred.probit[,params$burnin:params$nsweeps]),1,mean)

### XGBoost ###
xgboost_parms = list(booster = 'gbtree',objective = 'binary:logistic',silent=T,
eta=.1,min_child_weight = 10,colsample = 0.7)
t5 = proc.time()
dtrain = xgb.DMatrix(x,label=y)
bst = xgb.train(dtrain,params=xgboost_parms, nrounds = 200) # Fit XGBoost
yhat.train.xgb = predict(bst,newdata=x)
yhat.test.xgb = predict(bst,newdata=xtest) # Predict on test
t5 = proc.time() - t5
print(t5)


# print("BART: ")
# print(auc(ytest,phat))
print("XBART: ")
print(auc(ytest,yhat.pred))
print("XBART CLT: ")
print(auc(ytest,yhat.clt.pred))
# print("XBART Probit: ")
# print(auc(ytest,yhat.probit.pred))
print("XGBoost: ")
print(auc(ytest,yhat.test.xgb))

print("Importance:")
print("XBART: ")
print(out$importance)
print("XBART CLT: ")
print(out_clt$importance)
# print("XBART Probit: ")
# print(out_probit$importance)
print("Xgboost: ")

#plot(phat,yhat.pred)
#print("Done!")

plotROC(yhat.pred,ytest,col='orange')
plotROC(yhat.clt.pred ,ytest,add = TRUE, col='green')
plotROC(yhat.probit.pred ,ytest,add = TRUE, col='pink')
plotROC(yhat.test.xgb,ytest,add=TRUE,col="purple")
plotROC(phat,ytest,add=TRUE)


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
