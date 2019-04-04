library("dbarts")
library("XBART")

n = 1000
x1 = runif(n,-3,3)
x2 = runif(n,-3,3)

x = matrix(runif(n*5), ncol=5)
xtest = matrix(runif(n*5), ncol=5)
 
xtest1 = runif(n,-3,3)
xtest2 = runif(n,-3,3)

f <- function(x){5*sin(3*x[,1])+2*x[,2]^2 + 3*(x[,3]*x[,4])}
logit <- function(x){exp(x - mean(x))/(1+exp(x-mean(x)))}

y = (logit(f(x)+rnorm(n)) > 0.5)*1
ytest =(logit(f(xtest)+rnorm(n)) > 0.5)*1
#y = x1^2+x1 -10 >= -2*x2^2-x2 + rnorm(n)

#ytest = xtest1^2+xtest1 -10 >= -2*xtest2^2-xtest2 + rnorm(n)

#plot(x1*y,x2*y,pch=20)

# fit = bart(x,as.numeric(y),cbind(xtest),ndpost=5000)
# phat = colMeans(pnorm(fit$yhat.test))
# print(mean((ytest == (phat>0.5))))

### fit xbart ###
### Helpers
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 5,
    L = 1,
    nsweeps = 40,
    Nmin = 1,
    alpha = 0.95,
    beta = 5,
    mtry = 1,
    burnin = 0)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(50, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 50;XBART_params$tau = var(y)/num_tress
    return(XBART_params)
}
params = get_XBART_params(n,5,y)



out = XBART.pl(as.matrix(y), as.matrix(x), as.matrix(xtest), params$M, params$L, p_categorical = 0, params$nsweeps, params$max_depth,
    params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1,
    mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
    Ncutpoints = params$Ncutpoints, parallel = FALSE,a=1/5,b = 1/5)

 pred = predict(out,as.matrix(xtest))
 yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean)
 #mean((ytest == (phat>0.5)))
 print(mean((ytest == (yhat.pred>0.5))))

print("Done!")

