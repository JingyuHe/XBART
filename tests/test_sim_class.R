library(XBART)
library(xgboost)
library(caret)

n.train = 10000
n.test = 5000
n.all = n.train + n.test
k = 10
p = k
acc_level = 0.95
nthread = 6
x = matrix(rnorm(n.all*(p)),n.all,(p))


QR = qr(matrix(rnorm(p*p),p,p))

lam = diag(qr.R(QR))
Lam = diag(lam/abs(lam))
R = qr.Q(QR)%*%Lam

entropy = function(x){  1 / sum(exp(-x * ((0:(p-1))/(p-1))^2)) - acc_level}
kappa = uniroot(entropy, c(0, 1000), tol = 1e-5)$root

# for (i in 1:(p-2)){
#   R[i, (i+2):p] = 0
#   R[i+2, 1:i] = 0
# }
xprime = x[,1:p]%*%R
#xprime = x
D = as.matrix(dist(1:p),p,p)/(p-1)
# kappa = 18.38048
error.mat = exp(-kappa*D^2)


error.mat = error.mat%*%diag(1/colSums(error.mat))
# barplot(error.mat[,3])
ind = apply(xprime,1,which.max)

y.all = sapply(1:n.all,function(j) sample(1:p,1,prob = error.mat[,ind[j]]))

X_train = x[1:n.train,]
X_test = x[-(1:n.train),]

y_train = y.all[1:n.train] - 1
y_test = y.all[-(1:n.train)] - 1
true_label = ind[-(1:n.train)] - 1

# num_sweeps = 30
# burnin = 3
# max_depth = 30
# num_trees = 20 # c(10, 30)

num_class = max(y_train)+1

# num_sweeps = ceiling(200/log(n)) 
num_sweeps = 20
burnin = 3
num_trees = 20
max_depth = 10
mtry = ceiling(p/2) 
if (TRUE){

  w = 1
  hmult = 1; heps = 0
  tm = proc.time()
  fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test, 
                          num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                          num_cutpoints=NULL, alpha=0.95, beta=1.25, tau_a = 1, tau_b = 1, 
                          no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = 0, 
                          kap = 1, s = 1, verbose = TRUE, set_random_seed = FALSE, 
                          random_seed = NULL, sample_weights_flag = TRUE, separate_tree = TRUE, stop_threshold = 0, 
                          weight = w, hmult = hmult, heps = heps, update_tau = FALSE) 
  tm = proc.time()-tm
  cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
  a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
  a = a / rowSums(a)
  yhat = apply(a,1,which.max)-1
  
  
  w_cand = seq(0.8, 2, by = 0.02)
  ll = rep(0, length(w_cand))
  for (i in 1:length(w_cand))
  {
    w = w_cand[i]
    prob = a^w / rowSums(a^w)
    spr.xbart <- split(prob, row(prob))
    ll[i] = sum(mapply(function(x,y) -log(x[y]), spr.xbart, y_test+1, SIMPLIFY =TRUE))
  }
  cat(paste('best w ', w_cand[which.min(ll)], "\n"))
  logloss <- min(ll)
  
  
  cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
  cat(paste("xbart default logloss : ",round(ll[6],3)),"\n")
  cat(paste("xbart logloss : ",round(logloss,3)),"\n")
  
  # plot(w_cand, ll)
}

if (TRUE){
  xgb.cross_validate = function(X_train, y_train, X_test, y_test, nthread){
    
    max_depth = c(10, 20)
    subsample = c(0.8)
    eta = c(0.05) 
    min_child_weight = c(1, 15)
    colsample_bytree = c(0.5, 1)
    param_grid = expand.grid(max_depth = max_depth, subsample = subsample, eta = eta,
                             colsample_bytree = colsample_bytree,  min_child_weight = min_child_weight)
    
    fold = createDataPartition(y_train, 1, p = 0.2)
    # fold = createFolds(y_train, 3)
    CV_error = matrix(0, dim(param_grid)[1], length(fold))
    logloss = matrix(0, dim(param_grid)[1], length(fold))
    for (j in 1:dim(param_grid)[1]) {
      for (k in 1:length(fold)) {
        fit.xgb <- xgboost(data = as.matrix(X_train[-fold[[k]],]), label = matrix(y_train[-fold[[k]]]), nthread = nthread, 
                           num_class=num_class, verbose = 0, nrounds=500,
                           early_stopping_rounds = 50,
                           eta = param_grid$eta[j],
                           max_depth = param_grid$max_depth[j],
                           subsample = param_grid$subsample[j],
                           colsample_bytree = param_grid$colsample_bytree[j],
                           min_child_weight = param_grid$min_child_weight[j],
                           params=list(objective="multi:softprob"))
        
        phat.xgb <- predict(fit.xgb, as.matrix(X_train[fold[[k]],]))
        phat.xgb <- matrix(phat.xgb, ncol=num_class, byrow=TRUE)
        yhat.xgb <- max.col(phat.xgb) - 1
        CV_error[j, k] = mean(yhat.xgb == y_train[fold[[k]]])
        
        spr <- split(phat.xgb, row(phat.xgb))
        logloss[j, k] <- sum(mapply(function(x,y) -log(x[y]), spr, y_train[fold[[k]]]+1, SIMPLIFY =TRUE))
      }
    }
    
    best_acc = which.max(rowMeans(CV_error, na.rm = TRUE))
    best_logloss = which.min(rowMeans(logloss, na.rm = TRUE))
    print("xgboost best acc param: \n")
    print(param_grid[best_acc,])
    print("xgboost best logloss param: \n")
    print(param_grid[best_logloss, ])
    return(list(best_acc = param_grid[best_acc,], best_logloss = param_grid[best_logloss,]))
  }
  
  tm2 = proc.time()
  xgb_param = xgb.cross_validate(X_train, y_train, X_test, y_test, nthread)
  xgb_acc <- xgboost(data = as.matrix(X_train), label = matrix(y_train),
                     num_class=num_class, verbose = 0, nthread = nthread,
                     eta = xgb_param[[1]]$eta,
                     max_depth = xgb_param[[1]]$max_depth,
                     subsample = xgb_param[[1]]$subsample,
                     colsample_bytree = xgb_param[[1]]$colsample_bytree,
                     min_child_weight = xgb_param[[1]]$min_child_weight,
                     nrounds=500,
                     early_stopping_rounds = 50,
                     params=list(objective="multi:softprob"))
  phat.xgb_acc <- predict(xgb_acc, as.matrix(X_test))
  
  xgb_logloss <- xgboost(data = as.matrix(X_train), label = matrix(y_train),
                         num_class=num_class, verbose = 0, nthread = nthread, 
                         eta = xgb_param[[2]]$eta,
                         max_depth = xgb_param[[2]]$max_depth,
                         subsample = xgb_param[[2]]$subsample,
                         colsample_bytree = xgb_param[[2]]$colsample_bytree,
                         min_child_weight = xgb_param[[2]]$min_child_weight,
                         nrounds=500,
                         early_stopping_rounds = 50,
                         params=list(objective="multi:softprob"))
  phat.xgb_logloss <- predict(xgb_logloss, as.matrix(X_test))
  
  tm2 = proc.time() - tm2
  
  phat.xgb <- matrix(phat.xgb_acc, ncol=num_class, byrow=TRUE)
  yhat.xgb <- max.col(phat.xgb) - 1
  
  phat.xgb_logloss <- matrix(phat.xgb_logloss, ncol=num_class, byrow=TRUE)
  spr.xgb <- split(phat.xgb_logloss, row(phat.xgb_logloss))
  logloss.xgb <- sum(mapply(function(x,y) -log(x[y]), spr.xgb, y_test+1, SIMPLIFY =TRUE))
  
  
}


cat(paste("xbart logloss : ",round(logloss,3)),"\n")
cat(paste("xgboost logloss : ", round(logloss.xgb,3)),"\n")

cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")

yhat = apply(a,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("xgboost classification accuracy: ", round(mean(yhat.xgb == y_test),3)),"\n")


cat("importance ", fit$importance, "\n")

# par(mfrow = c(1, 2))
# plot(as.vector(fit$weight))
# plot(as.vector(fit$tau_a))
summary(as.vector(fit$weight))
# plot(w_cand, ll)
