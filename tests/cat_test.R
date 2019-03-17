library(XBART)
library(xgboost)
library(dbarts)
library(ranger)
# generate data
set.seed(1933)
d = 5
n = 5000
nt = 5000
old_rmse = rep(NA,100)
new_rmse = rep(NA,100)

for(i in 1:100){

  print(paste0("On iteration ", i))
if(1){
  prob = runif(d,0.2,0.8)
  x = matrix(0,n,d+1)
  x[,1] = ceiling(abs(rnorm(n,25,10)))
  #x[,2] = rnorm(n,25,10)
  for (h in 1:d){
  x[,h+1] = -rbinom(n,1,prob[h])
  }
  
  
  xtest = matrix(0,nt,d+1)
  xtest[,1] = ceiling(abs(rnorm(nt,25,10)))
  #xtest[,2] = rnorm(n,25,10)
  for (h in 1:d){
    xtest[,h+1] = rbinom(nt,1,prob[h])
  }
  
  
  f = function(x){
    
level =  15 - 20*(x[,1]-25)^2/1500
level = level + 15*(x[,2] & x[,3]) -10*(x[,5] | x[,6])
level = level*(2*x[,4]-1)
  }
 
  
  ftrue = f(x)
  ftest = f(xtest)
  sigma = 3*sd(ftrue)
  
  
  
  y = ftrue + sigma*rnorm(n)
    
}


#####################################################
# # BART
# if (1){
#  par(mfrow = c(2,2))
#  hist(y,40)
#  t1 = proc.time()
#  fit = bart(x,y,xtest,ndpost = 1000, nskip = 1000)
#  t1 = proc.time() - t1
#  plot(ftest, fit$yhat.test.mean,pch=20, main = "BART")
#  abline(0,1)

#  RMSE1 = sqrt(mean((ftest - fit$yhat.test.mean)^2))
# }

#####################################################
# XBART

ntrue = n
mtry = min(floor(sqrt(d)),20)
Ncutpoints = floor(sqrt(n))
#Ncutpoints = max(Ncutpoints,100)

L = 1
#M = 3*floor(sqrt(n)) # number of trees
M = floor(0.25*log(n)^(log(log(n))))
K = 0
ntrees = M+K
print(M)
print(K)
nsweeps = 40
burnin = 15
Nmin = 1
print(c(n,d,Ncutpoints,Nmin))


# max_depth = matrix(rep(1:nsweeps, M), M, nsweeps, byrow= TRUE) + 1

max_depth = matrix(250, ntrees, nsweeps)

#max_depth = matrix(rpois(ntrees*nsweeps,30),ntrees,nsweeps)
#for (j in 1:2){ 
#  max_depth[,j] = 5}


alpha = 0.95
beta = 1.25
#tau = 0.05
#tau = 2*var(y)/(M)

tau = (3/10)*var(y)/ntrees # var(y)*E(1/(total #of terminal nodes))

yhats = rep(0,n)
yhat.train = rep(0,n)

yhat.test = rep(0,nt)
yhat.test2 = rep(0,nt)

mc = 1

t2 = proc.time()
# fit = train_forest_root_std_mtrywithinnode_ordinal(as.matrix(y), as.matrix(x), as.matrix(xtest), M, L, nsweeps, max_depth, Nmin, alpha = alpha, beta = beta, tau = tau, s= 1,kap = 1, mtry = mtry, burnin = burnin,m_update_sigma = TRUE,draw_mu= TRUE, Ncutpoints = Ncutpoints, parallel = FALSE)


cat("----------------------------------------- \n")
cat("----------------------------------------- \n")


fit = train_forest_root_std_all(as.matrix(y), as.matrix(x), as.matrix(xtest), M, L, nsweeps, max_depth, Nmin, alpha = alpha, beta = beta, p_categorical = dim(x)[2], tau = tau, s= 1,kap = 1, mtry = mtry, burnin = burnin, m_update_sigma = TRUE,draw_mu= TRUE, Ncutpoints = Ncutpoints, parallel = FALSE)


t2 = proc.time() - t2
  
  
# t4 = proc.time() - t4
yhatLL.test = apply(fit$yhats_test[,burnin:nsweeps],1,mean)
yhatLL.train = apply(fit$yhats[,burnin:nsweeps],1,mean)
rmseLL = sqrt(mean((ftest - yhatLL.test)^2))


#plot(ftest, yhatLL.test,pch=20, main = "XBART")
#abline(0,1)



RMSE2 = sqrt(mean((ftest - yhatLL.test)^2))

t3 = proc.time()
fit2 = XBART(as.matrix(y), as.matrix(x), as.matrix(xtest), M, L, nsweeps, max_depth, Nmin, alpha = alpha, beta = beta, p_categorical = dim(x)[2], tau = tau, s= 1,kap = 1, mtry = mtry, burnin = burnin, m_update_sigma = TRUE,draw_mu= TRUE, Ncutpoints = Ncutpoints, parallel = FALSE)
t3 = proc.time() - t3


yhatLL.test_2 = apply(fit2$yhats_test[,burnin:nsweeps],1,mean)
yhatLL.train_2 = apply(fit2$yhats[,burnin:nsweeps],1,mean)
RMSE3 = sqrt(mean((ftest - yhatLL.test_2)^2))

RMSE_OF_FITS = sqrt(mean((yhatLL.test - yhatLL.test_2)^2))

cat("RMSE of XBART Orignal ", RMSE2, " running time ", t2, "\n")
cat("RMSE of XBART Refactor ", RMSE3, " running time ", t3, "\n")
cat("RMSE of fits: ",RMSE_OF_FITS, "\n")
old_rmse[i] = RMSE2
new_rmse[i] = RMSE3
}

cat("Mean RMSE of XBART Orignal ", mean(old_rmse), " running time ", t2, "\n")
cat("Mean RMSE of XBART Refactor ", mean(new_rmse), " running time ", t2, "\n")
cat("Percentage of time NEW RMSE higher than OLD ", mean(new_rmse>old_rmse) , "\n")

cat("Var RMSE of XBART Orignal ", var(old_rmse), " running time ", t2, "\n")
cat("Var RMSE of XBART Refactor ", var(new_rmse), " running time ", t2, "\n")

