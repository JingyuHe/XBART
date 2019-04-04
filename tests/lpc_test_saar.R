library("dbarts")
library("XBART")

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



n = 1000
x1 = runif(n,-2,2)
x2 = runif(n,-2,2)

 
xtest1 = runif(n,-2,2)
xtest2 = runif(n,-2,2)

f = function(x1,x2,error){
  
  abs(x1)/(abs(x2)+1) >= abs(x2)/(abs(x1)+1) + error
  
 # x1^2 + x2 - 1 >= x2^2 - 2*x1 + error
  
}

y = f(x1,x2,runif(n,-0.5,0.5))

ytest = f(xtest1,xtest2,runif(n,-0.5,0.5))

plot(x1*y,x2*y,pch=20)

fit = bart(cbind(x1,x2),as.numeric(y),cbind(xtest1,xtest2),ndpost=1000)
phat = colMeans(pnorm(fit$yhat.test))
mean((ytest == (phat>0.5)))

### fit xbart ###
### Helpers
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 20,
    L = 1,
    nsweeps = 200,
    Nmin = 2,
    alpha = 0.95,
    beta = 1.25,
    mtry = 2,
    burnin = 20)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(350, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 150;XBART_params$tau = 100
    XBART_params$a = 0.01/num_tress; XBART_params$b = 0.01/num_tress;
    return(XBART_params)
}
params = get_XBART_params(n,2,y)



out = XBART.pl(as.matrix(y), as.matrix(cbind(x1,x2)), as.matrix(cbind(x1,x2)), params$M, params$L, p_categorical = 0, params$nsweeps, params$max_depth,
    params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1,
    mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
    Ncutpoints = params$Ncutpoints, parallel = FALSE,a=params$a,b = params$b)

 pred = predict(out,as.matrix(cbind(xtest1,xtest2)))
 yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean)
print(mean((ytest == (phat>0.5))))
print(mean((ytest == (yhat.pred>0.5))))
plot(phat,yhat.pred)
print("Done!")

plotROC(phat,ytest)
plotROC(yhat.pred,ytest,add=TRUE,col='orange')
