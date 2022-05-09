library(XBART)
library(xgboost)

n.train = 5000
n.test = 1000
n.all = n.train + n.test
k = 3
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
num_trees = 50
max_depth = 10
mtry = (p+p_add)/2 # round((p + p_cat)/3)
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                        num_cutpoints=NULL, burnin = burnin, mtry = mtry, p_categorical = p_cat, 
                        verbose = FALSE, separate_tree = FALSE, updte_tau = FALSE) 
tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
phat = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
yhat = apply((phat),1,which.max)-1

spr.xbart <- split(phat, row(phat))
logloss <-sum(mapply(function(x,y) -log(x[y]), spr.xbart, y_test+1, SIMPLIFY =TRUE))

cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("xbart logloss : ",round(logloss,3)),"\n")

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

spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

cat(paste("xbart logloss : ",round(logloss,3)),"\n")
# cat(paste("xgboost logloss : ", round(logloss.xgb,3)),"\n")

cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")

cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("xgboost classification accuracy: ", round(mean(yhat.xgb == y_test),3)),"\n")
