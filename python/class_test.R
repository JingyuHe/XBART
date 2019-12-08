library("XBART")


data(iris)

y <- matrix(0, nrow = nrow(iris), ncol = 3)
for (i in 1:nrow(iris)){
  y[i, as.numeric(iris$Species)[i]] <- 1
}

X <- model.matrix(~.-Species-1, data = iris)

num_trees <- 100
num_sweeps <- 100
max_depth <- 10

rslt <- XBART.Probit(y = y, X = X, Xtest =X,
             num_trees = num_trees,
             num_sweeps = num_sweeps,
             max_depth = 10, 
             Nmin = 2,
             num_cutpoints = 10,
             alpha = 1,
             beta = 1, 
             tau = 1)
