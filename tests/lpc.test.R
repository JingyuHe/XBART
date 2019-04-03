library("dbarts")
library("XBART")

n = 1000
x1 = runif(n,-3,3)
x2 = runif(n,-3,3)

 
xtest1 = runif(n,-3,3)
xtest2 = runif(n,-3,3)

y = x1^2+x1 -10 >= -2*x2^2-x2 + rnorm(n)

ytest = xtest1^2+xtest1 -10 >= -2*xtest2^2-xtest2 + rnorm(n)

plot(x1*y,x2*y,pch=20)

#fit = bart(cbind(x1,x2),as.numeric(y),cbind(xtest1,xtest2),ndpost=5000)
#phat = colMeans(pnorm(fit$yhat.test))
#mean((ytest == (phat>0.5)))

### fit xbart ###
### Helpers
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 10,
    L = 1,
    nsweeps = 40,
    Nmin = 1,
    alpha = 0.95,
    beta = 1.25,
    mtry = 1,
    burnin = 15)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 50;XBART_params$tau = var(y)/num_tress
    return(XBART_params)
}
params = get_XBART_params(n,2,y)
for(i in 1:50){
    print(i)
    XBART.pl(as.matrix(y), as.matrix(cbind(x1,x2)), as.matrix(cbind(x1,x2)), params$M, params$L, p_categorical = 0, params$nsweeps, params$max_depth,
    params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1,
    mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
    Ncutpoints = params$Ncutpoints, parallel = FALSE,a=1/params$M,b = 1/params$M)


}
print("Done!")

