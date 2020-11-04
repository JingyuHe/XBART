library(XBART)
library(xgboost)

n.train = 10000
n.test = 5000
n.all = n.train + n.test
k = 10
p = k
p_add = 0
p_cat = 0
acc_level = 0.5
x = matrix(rnorm(n.all*(p+p_add)),n.all,(p+p_add))


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

# num_sweeps = 30
# burnin = 3
# max_depth = 30
# num_trees = 20 # c(10, 30)

num_class = max(y_train)+1

# num_sweeps = ceiling(200/log(n)) 
num_sweeps = 20
burnin = 3
num_trees = 150
max_depth = 20
mtry = (p+p_add)/2 # round((p + p_cat)/3)
if (TRUE){
  # w_cand = seq(5, 15, by = 2)
  # acc = rep(0, length(w_cand))
  # #########################  parallel ####################3
  # for (i in 1:length(w_cand))
  # {
  #   tm = proc.time()
  #   fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test,
  #                           num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth,
  #                           num_cutpoints=NULL, alpha=0.95, beta=1.25, tau_a = 1, tau_b = 1,
  #                           no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = p_cat,
  #                           kap = 1, s = 1, verbose = FALSE, set_random_seed = FALSE,
  #                           random_seed = NULL, sample_weights_flag = TRUE, separate_tree = FALSE, stop_threshold = 0,
  #                           weight = w_cand[i], hmult = 1, heps = 0)
  #   tm = proc.time()-tm
  #   cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
  #   a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
  #   yhat = apply(a,1,which.max)-1
  #   acc[i] = round(mean(y_test == yhat),3)
  # }
  # cat(paste('best w ', w_cand[which.max(acc)], "\n"))
  # cat(paste("xbart classification accuracy: ",round(max(acc),3)),"\n")
  # 
  # w = w_cand[which.min(acc)]
  w = 1
  hmult = 1; heps = 0
  tm = proc.time()
  fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test, 
                          num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                          num_cutpoints=NULL, alpha=0.95, beta=1.25, tau_a = 1, tau_b = 1, 
                          no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = p_cat, 
                          kap = 1, s = 1, verbose = FALSE, set_random_seed = FALSE, 
                          random_seed = NULL, sample_weights_flag = TRUE, separate_tree = FALSE, stop_threshold = 0, 
                          weight = w, hmult = hmult, heps = heps) 
  tm = proc.time()-tm
  cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
  a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
  yhat = apply(a,1,which.max)-1
  
  
  w_cand = seq(0.1, 3, by = 0.05)
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


tm2 = proc.time()
fit.xgb <- xgboost(data = X_train, label=y_train,
                   num_class=k,
                   verbose = 0,
                   max_depth = 4,
                   subsample = 0.80,
                   nrounds=500,
                   early_stopping_rounds = 2,
                   eta = 0.1,
                   params=list(objective="multi:softprob"))

tm2 = proc.time()-tm2
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol=k, byrow=TRUE)

yhat.xgb <- max.col(phat.xgb) - 1


pr = sapply(1:n.test, function(i) error.mat[ y_test[i] + 1, y_test[i] + 1])
cat(paste("xbart rmse on probabilities: ", round(sqrt(mean((a-pr)^2)),3)),"\n")
cat(paste("xgboost rmse on probabilities: ", round(sqrt(mean((phat.xgb-pr)^2)),3)),"\n")

# spr <- split(a, row(a))
# logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))
spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

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
