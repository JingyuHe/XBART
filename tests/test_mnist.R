library(XBART)
library(xgboost)

D <- read.csv('~/Dropbox/MNIST/mnist_train.csv',header=FALSE)
y = D[,1]
D = D[,-1]
# 
Dtest <- read.csv('~/Dropbox/MNIST/mnist_test.csv',header=FALSE)
ytest = Dtest[,1]
Dtest = Dtest[,-1]
pred = matrix(0,10000,10)

X_train = D
X_test = Dtest
p = ncol(X_train)

for (i in 1:p){
  X_train[, i] = X_train[, i] + 0.01*rnorm(length(y))
  X_test[, i] = X_test[, i] + 0.01*rnorm(length(ytest))
}

# if (!exists("num_trees")){num_trees = 120}
# if (!exists("num_sweeps")){num_sweeps = 40}
# if (!exists("burnin")){burnin = 15}
# # if (!exists("delta")){delta = seq(0.1, 2, 0.05)}
# # if (!exists("concn")){concn = 1}
# if (!exists("Nmin")){Nmin = 1}
# if (!exists("max_depth")){max_depth = 10}
# if (!exists("mtry")){mtry = 10}
# if (!exists("num_cutpoints")){num_cutpoints = 20}
num_sweeps= 15
num_trees = 10
burnin = 2
Nmin = 5
max_depth = 25
mtry = 50
num_cutpoints=20

tau = 100 / num_trees
tau_later = 100 / num_trees


t = proc.time()
fit = XBART.multinomial(y=matrix(y), num_class=10, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                        Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau=100/num_trees, 
                        no_split_penality = 1, weight = c(1), burnin = burnin, mtry = mtry, p_categorical = 0L, 
                        kap = 1, s = 1, verbose = TRUE, parallel = FALSE, set_random_seed = TRUE, 
                        random_seed = NULL, sample_weights_flag = TRUE, separate_trees = TRUE, sample_var_per_tree = FALSE, 
                        phi_threshold = 0.2) 
t = proc.time() - t


pred = apply(fit$yhats_test[(burnin+1):(num_sweeps-0),,], c(2,3), mean)
yhat = max.col(pred)-1

spr <- split(pred, row(pred))
logloss <- sum(unlist(mapply(function(x,y) -log(x[y]), spr, ytest, SIMPLIFY =TRUE)))

cat("running time ", t[3], " seconds \n")

cat("XBART error rate ", mean(yhat != ytest), "\n")

cat(paste("xbart logloss : ",round(logloss,3)),"\n")


for(i in 0:9){
  cat("XBART error rate in ", i, ": ", round(mean(yhat[ytest==i]!=i), 4), 
      " misclassified as ", tail(names(sort(table(yhat[ytest==i]))), 2)[1], "\n " )
}