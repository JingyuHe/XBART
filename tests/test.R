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
  XBART_params$max_depth = matrix(50, num_tress, XBART_params$nsweeps)
  XBART_params$Ncutpoints = max(3*floor(sqrt(n)),250);XBART_params$tau = 0.2*var(y)/(0.67*num_tress)
  return(XBART_params)
}


library(XBART)

d = 12
n = 10000
nt = 10000

x = matrix(runif(d*n,-2,2),n,d)
  
xtest = matrix(runif(d*nt,-2,2),nt,d)

f = function(x){
    
    1 + 3*apply(x[,1:3],1,max)
    
 } 

ftrue = f(x)
ftest = f(xtest)
sigma = 0.1*sd(ftrue)

y = ftrue + sigma*rnorm(n)
y_test = ftest + sigma*rnorm(nt)

params = get_XBART_params(n,d,y)


fit2 = train_forest_root_std_all(as.matrix(y), as.matrix(x), as.matrix(xtest), params$M, params$L, params$nsweeps, params$max_depth, 
                                                  params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
                                                  mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
                                                  Ncutpoints = params$Ncutpoints, parallel = FALSE)
yhat.2 = apply(fit2$yhats_test[,params$burnin:params$nsweeps],1,mean)

before = as.matrix(xtest)
fit = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), params$M, params$L, params$nsweeps, params$max_depth, 
                                                  params$Nmin, alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1, 
                                                  mtry = params$mtry, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE, 
                                                  Ncutpoints = params$Ncutpoints, parallel = FALSE,random_seed=100)
yhat.1 = apply(fit$yhats_test[,params$burnin:params$nsweeps],1,mean)

pred = XBART.predict(fit,as.matrix(xtest))
yhat.pred = apply(pred$yhats[,params$burnin:params$nsweeps],1,mean)
after = as.matrix(xtest)
print(all(before == after))




print(paste("rmse of fits: ",sqrt((mean(yhat.1-yhat.2)^2))))
print(paste("rmse of fit new predict: ",sqrt((mean(yhat.pred-y_test)^2))))
print(paste("rmse of fit new: ",sqrt((mean(yhat.1-y_test)^2))))
print(paste("rmse of fit old: ",sqrt((mean(yhat.2-y_test)^2))))




