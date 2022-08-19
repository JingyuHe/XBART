###################################################
# This script shows regression using XBART
###################################################



#######################################################################
# set parameters of XBART
get_XBART_params <- function(y) {
    XBART_params <- list(
        num_trees = 30, # number of trees
        num_sweeps = 40, # number of sweeps (samples of the forest)
        n_min = 1, # minimal node size
        alpha = 0.95, # BART prior parameter
        beta = 1.25, # BART prior parameter
        mtry = 10, # number of variables sampled in each split
        burnin = 15,
        no_split_penality = "Auto"
    ) # burnin of MCMC sample
    num_tress <- XBART_params$num_trees
    XBART_params$max_depth <- 250
    XBART_params$num_cutpoints <- 50
    # number of adaptive cutpoints
    XBART_params$tau <- var(y) / num_tress # prior variance of mu (leaf parameter)
    return(XBART_params)
}


#######################################################################
library(XBART)

set.seed(100)
new_data <- TRUE # generate new data
run_dbarts <- FALSE # run dbarts
run_xgboost <- FALSE # run xgboost
run_lightgbm <- FALSE # run lightgbm
parl <- FALSE # parallel computing


small_case <- TRUE # run simulation on small data set
verbose <- FALSE # print the progress on screen


if (small_case) {
    n <- 20000 # size of training set
    nt <- 5000 # size of testing set
    d <- 20 # number of TOTAL variables
    dcat <- 10 # number of categorical variables
    # must be d >= dcat
    # (X_continuous, X_categorical), 10 and 10 for each case, 20 in total
} else {
    n <- 100000
    nt <- 10000
    d <- 30
    dcat <- 0
}






#######################################################################
# Data generating process

#######################################################################
# Have to put continuous variables first, then categorical variables  #
# X = (X_continuous, X_cateogrical)                                   #
#######################################################################
if (new_data) {
    if (d != dcat) {
        x <- matrix(runif((d - dcat) * n, -2, 2), n, d - dcat)
        if (dcat > 0) {
            x <- cbind(x, matrix(as.numeric(sample(-2:2, dcat * n, replace = TRUE)), n, dcat))
        }
    } else {
        x <- matrix(as.numeric(sample(-2:2, dcat * n, replace = TRUE)), n, dcat)
    }


    if (d != dcat) {
        xtest <- matrix(runif((d - dcat) * nt, -2, 2), nt, d - dcat)
        if (dcat > 0) {
            xtest <- cbind(xtest, matrix(as.numeric(sample(-2:2, dcat * nt, replace = TRUE)), nt, dcat))
        }
    } else {
        xtest <- matrix(as.numeric(sample(-2:2, dcat * nt, replace = TRUE)), nt, dcat)
    }

    f <- function(x) {
        sin(rowSums(x[, 3:4]^2)) + sin(rowSums(x[, 1:2]^2)) + (x[, 15] + x[, 14])^2 * (x[, 1] + x[, 2]^2) / (3 + x[, 3] + x[, 14]^2)
        # rowSums(x[,1:30]^2)
        # pmax(x[,1]*x[,2], abs(x[,3])*(x[,10]>x[,15])+abs(x[,4])*(x[,10]<=x[,15]))
        #
    }

    # to test if ties cause a crash in continuous variables
    x[, 1] <- round(x[, 1], 4)
    # xtest[,1] = round(xtest[,1],2)
    ftrue <- f(x)
    ftest <- f(xtest)
    sigma <- sd(ftrue)

    # y = ftrue + sigma*(rgamma(n,1,1)-1)/(3+x[,d])
    # y_test = ftest + sigma*(rgamma(nt,1,1)-1)/(3+xtest[,d])

    y <- ftrue + sigma * rnorm(n)
    y_test <- ftest + sigma * rnorm(nt)
}

#######################################################################
# XBART
categ <- function(z, j) {
    q <- as.numeric(quantile(x[, j], seq(0, 1, length.out = 100)))
    output <- findInterval(z, c(q, +Inf))
    return(output)
}

nthread = 8

params <- get_XBART_params(y)
time <- proc.time()
fit <- XBART(as.matrix(y), as.matrix(x), p_categorical = dcat, params$num_trees, params$num_sweeps, params$max_depth, params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1, mtry = params$mtry, verbose = verbose, num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100, no_split_penality = params$no_split_penality, nthread = nthread)
time <- proc.time() - time
cat("Running time ", time[3], " seconds.\n")

################################
# predict for testing set
pred <- predict(fit, xtest)
pred <- rowMeans(pred[, params$burnin:params$num_sweeps])


xbart_rmse = sqrt(mean((pred - ftest) ^ 2))
print(paste("rmse of fit xbart: ", round(xbart_rmse, digits = 4)))

plot(ftest, pred)
abline(0,1)

# predict with Gaussian process extrapolation for out-of-range data points
gp_pred <- predict_gp(fit, as.matrix(y), as.matrix(x), as.matrix(xtest))