library(XBART)
library(BART)
library(ranger)


seed = 10


set.seed(seed)


n = 7000
n_train = 6000

p = 2
X = matrix(runif(n*p), nrow=n)
logodds = -1 + 2*X[,1] + X[,2]^2 #- 0.5*X[,1]*X[,2]
# logodds = X[,1] + X[,2]
pr = plogis(logodds)
y = rbinom(n, 1, pr)

y_train = y[1:n_train]
y_test = y[(n_train + 1):n]
X_train = X[1:n_train, ]
X_test = X[(n_train + 1):n, ]

num_sweeps = 30
burnin = 20


if(0){
# insample error 
y_test = y_train
X_test = X_train
}else{

}


fit = XBART.multinomial(y=y_train, num_class=2, X=X_train, Xtest=X_test, 
            num_trees=100, num_sweeps=num_sweeps, max_depth=300, 
            Nmin=5, num_cutpoints=50, alpha=0.95, beta=1.25, tau=1, 
            no_split_penality = 0.5, burnin = 1L, mtry = 0L, p_categorical = 0L, 
            kap = 16, s = 4, verbose = FALSE, parallel = TRUE, set_random_seed = TRUE, 
            random_seed = seed, sample_weights_flag = TRUE) 

# number of sweeps * number of observations * number of classes
dim(fit$yhats_test)


# take average of all sweeps, discard burn-in
a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)


# final predcition
pred = as.numeric(a[,1] < a[,2])


# Compare with BART probit
fit2 = pbart(X_train, y_train)

pred2 = predict(fit2, X_test)
pred2 = as.numeric(pred2$prob.test.mean > 0.5)



# Compare with ranger
data = data.frame( y = y_train, X = X_train)
data.test = data.frame(X = X_test)

fit3 = ranger(y ~ ., data = data)

pred3 = predict(fit3, data.test)

pred3 = as.numeric(pred3$predictions > 0.5)



# OUT SAMPLE accuracy
sum(pred == y_test)
sum(pred2 == y_test)
sum(pred3 == y_test)
