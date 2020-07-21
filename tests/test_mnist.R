library(XBART)
library(xgboost)
# path = '~/Dropbox/MNIST/'
path = '~/mnist/'

D <- read.csv(paste(path,'mnist_train.csv', sep=''),header=FALSE)
y = D[,1]
D = D[,-1]
# 
Dtest <- read.csv(paste(path, 'mnist_test.csv', sep =''),header=FALSE)
ytest = Dtest[,1]
Dtest = Dtest[,-1]
pred = matrix(0,10000,10)

X_train = D
X_test = Dtest
p = ncol(X_train)

# for (i in 1:p){
#   X_train[, i] = X_train[, i] + 0.01*rnorm(length(y))
#   X_test[, i] = X_test[, i] + 0.01*rnorm(length(ytest))
# }


X.train = X_train
X.test = X_test

v = 0
for (h in 1:p){
  breaks =unique(as.numeric(quantile(c(X_train[,h],X_test[,h]),seq(0,1,length.out=4))))
  #breaks = seq(min(c(X_train[,h],X_test[,h])),max(c(X_train[,h],X_test[,h])),length.out = 25)
  breaks = c(-Inf,breaks,+Inf)
  #print(breaks)
  if (length(breaks)>3){
    v = v + 1
    X.train[,v] = cut(X_train[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
    X.test[,v] = cut(X_test[,h],breaks = breaks,include.lowest=TRUE,labels=FALSE)
  }
}

#print(v)
X_train = X.train[,1:v]
X_test = X.test[,1:v]
p = v



num_sweeps= 20
num_trees = 20
burnin = 5 #10
Nmin = 10
max_depth = 20
mtry = round(p/5)
num_cutpoints=20

drop_threshold = 1

# ws = seq(1, 15, 0.5)


##################### test run to drop variables #################
# t = proc.time()
# fit_test = XBART.multinomial(y=matrix(y), num_class=10, X=X_train, Xtest=X_test, 
#                         num_trees=num_trees, num_sweeps=3, max_depth=max_depth, 
#                         Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau_a = 2, tau_b = 2,
#                         no_split_penality = 1, weight = c(1), #seq(1, 10, 0.5), 
#                         burnin = 1, mtry = mtry, p_categorical = p, 
#                         kap = 1, s = 1, verbose = TRUE, parallel = TRUE, set_random_seed = TRUE, 
#                         random_seed = NULL, sample_weights_flag = TRUE, sample_per_tree = TRUE, stop_threshold = 0.1) 
# t = proc.time() - t
# cat("test fit running time ", t[3], " seconds \n")

# fit_test$importance

# X_train = X_train[, -which(fit_test$importance < drop_threshold)]
# X_test = X_test[, -which(fit_test$importance < drop_threshold)]
# p = ncol(X_train)
# cat('dropped variables ', which(fit_test$importance < drop_threshold) )
# #################################################################


###################### parallel #################
t = proc.time()
fit = XBART.multinomial(y=matrix(y), num_class=10, X=X_train, Xtest=X_test, 
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth, 
                        Nmin=Nmin, num_cutpoints=num_cutpoints, alpha=0.95, beta=1.25, tau_a = 1, tau_b = 1, 
                        no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = p, 
                        kap = 1, s = 1, verbose = TRUE, parallel = TRUE, set_random_seed = TRUE, 
                        random_seed = NULL, sample_weights_flag = TRUE, stop_threshold = 0, nthread = 0, weight = 1,
                        hmult = 1, heps = 0.1) 
t = proc.time() - t


pred = apply(fit$yhats_test[(burnin):(num_sweeps-0),,], c(2,3), mean)
yhat = max.col(pred)-1

spr <- split(pred, row(pred))
logloss <- sum(unlist(mapply(function(x,y) -log(x[y]), spr, ytest, SIMPLIFY =TRUE)))

cat("running time ", t[3], " seconds \n")

cat("XBART error rate ", mean(yhat != ytest), "\n")

cat(paste("xbart logloss : ",round(logloss,3)),"\n")
##################################################################



# for(i in 0:9){
#   cat("XBART error rate in ", i, ": ", round(mean(yhat[ytest==i]!=i), 4), 
#       " misclassified as ", tail(names(sort(table(yhat[ytest==i]))), 2)[1], "\n " )
# }
# 
saveRDS(fit, paste(path, 'mnist_result/logloss_072101.rds', sep = ''))
