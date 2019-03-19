### Helpers
get_XBART_params <- function(n,d,y){
  XBART_params = list(M = floor(0.5*log(n)^(log(log(n)))),
                      L = 1,
                      nsweeps = 50,
                      Nmin = 1,
                      alpha = 0.95,
                      beta = 1.25,
                      mtry = 10,
                      burnin = 15)
  num_tress = XBART_params$M
  XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
  XBART_params$Ncutpoints = max(3*floor(sqrt(n)),250);XBART_params$tau = var(y)/num_tress
  return(XBART_params)
}


library(XBART)
library(dbarts)

d = 10
dcat = 10
n = 500
nt = 500

if (d != dcat){x = matrix(runif((d-dcat)*n,-2,2),n,d-dcat)
if (dcat > 0){
  x = cbind(x,matrix(as.numeric(sample(-2:2,dcat*n,replace=TRUE)),n,dcat))}
}else{
  x = matrix(as.numeric(sample(-2:2,dcat*n,replace=TRUE)),n,dcat)}


if (d != dcat){xtest = matrix(runif((d-dcat)*nt,-2,2),nt,d-dcat)
if (dcat>0){
  xtest = cbind(xtest,matrix(as.numeric(sample(-2:2,dcat*nt,replace=TRUE)),nt,dcat))
}
}else{
  xtest = matrix(as.numeric(sample(-2:2,dcat*nt,replace=TRUE)),nt,dcat)}


f = function(x){
  
  x[,8] + 3*(x[,2]>x[,6]) + 2*(x[,7]<x[,5])
  
} 

ftrue = f(x)
ftest = f(xtest)
sigma = 0.2*sd(ftrue)

y = ftrue + sigma*rnorm(n)
y_test = ftest + sigma*rnorm(nt)

params = get_XBART_params(n,d,y)
time = proc.time()
fit = bart(x,y,xtest,verbose = FALSE)
time = proc.time() - time
print(time[3])

fhat.db = fit$yhat.test.mean


#fit2 = train_forest_root_std_all(as.matrix(y), as.matrix(x), as.matrix(xtest), params$M, params$L, p_categorical = dcat, params$nsweeps, params$max_depth, 
  #                               params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
  #                               mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
  #                               Ncutpoints = params$Ncutpoints, parallel = FALSE)
#fhat.2 = apply(fit2$yhats_test[,params$burnin:params$nsweeps],1,mean)

#x = x+runif(length(x),-0.01,0.01)
#dcat = 0

time = proc.time()
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), p_categorical = dcat, params$M, params$L, params$nsweeps, params$max_depth, 
            params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
            mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
            Ncutpoints = params$Ncutpoints, parallel = FALSE)
time = proc.time()-time
print(time[3])

fhat.1 = apply(fit$yhats_test[,params$burnin:params$nsweeps],1,mean)
#print(yhat.1[0:10])


print(paste("rmse of fit xbart: ",sqrt(mean((fhat.1-ftest)^2))))
#print(paste("rmse of fit old: ",sqrt(mean((fhat.2-ftest)^2))))
print(paste("rmse of fit dbart: ",sqrt(mean((fhat.db-ftest)^2))))

  plot(ftest,fhat.1,pch=20,col='slategray')
  points(ftest,fhat.db,pch=20,col='orange')

