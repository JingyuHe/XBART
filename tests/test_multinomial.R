library(XBART)

n = 1000
p = 2
X = matrix(runif(n*p), nrow=n)
logodds = -1 + 2*X[,1] + X[,2]^2 - 0.5*X[,1]*X[,2]
pr = plogis(logodds)
y = rbinom(n, 1, pr)

XBART_multinomial(y=y, num_class=2, X=X, Xtest=X, 
                  num_trees=100, num_sweeps=40, max_depth=300, 
                  n_min=5, num_cutpoints=50, alpha=0.95, beta=1.25, tau=1, 
                  no_split_penality = 0.5, burnin = 1L, mtry = 0L, p_categorical = 0L, 
                  kap = 16, s = 4, verbose = TRUE, parallel = TRUE, set_random_seed = FALSE, 
                  random_seed = 0L, sample_weights_flag = TRUE) 
