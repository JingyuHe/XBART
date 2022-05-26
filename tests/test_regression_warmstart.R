###################################################
# This script demonsrates warmstart BART package with XBART trees
# regression case
# use the customized BART package https://github.com/jingyuhe/BART
###################################################



#######################################################################
# set parameters of XBART
get_XBART_params <- function(y) {
    XBART_params <- list(
        num_trees = 30, # number of trees
        num_sweeps = 50, # number of sweeps (samples of the forest)
        n_min = 1, # minimal node size
        alpha = 0.95, # BART prior parameter
        beta = 1.25, # BART prior parameter
        mtry = 10, # number of variables sampled in each split
        burnin = 15,
        no_split_penality = "Auto"
    ) # burnin of MCMC sample
    num_tress <- XBART_params$num_trees
    XBART_params$max_depth <- 250
    XBART_params$num_cutpoints <- 100
    # number of adaptive cutpoints
    XBART_params$tau <- var(y) / num_tress # prior variance of mu (leaf parameter)
    return(XBART_params)
}


#######################################################################
library(XBART)
library(BART)

set.seed(100)
new_data <- TRUE # generate new data
run_dbarts <- FALSE # run dbarts
run_xgboost <- FALSE # run xgboost
run_lightgbm <- FALSE # run lightgbm
parl <- TRUE # parallel computing


small_case <- TRUE # run simulation on small data set
verbose <- FALSE # print the progress on screen


if (small_case) {
    n <- 10000 # size of training set
    nt <- 5000 # size of testing set
    d <- 20 # number of TOTAL variables
    dcat <- 0 # number of categorical variables
    # must be d >= dcat
    # (X_continuous, X_categorical), 10 and 10 for each case, 20 in total
} else {
    n <- 100000
    nt <- 10000
    d <- 50
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
params <- get_XBART_params(y)
time <- proc.time()

nthread <- 4

# XBART
fit <- XBART(as.matrix(y), as.matrix(x), p_categorical = dcat, params$num_trees, params$num_sweeps, params$max_depth, params$n_min, alpha = params$alpha, beta = params$beta, tau = params$tau, s = 1, kap = 1, mtry = params$mtry, verbose = verbose, num_cutpoints = params$num_cutpoints, parallel = parl, random_seed = 100, no_split_penality = params$no_split_penality, nthread = nthread)

time <- proc.time() - time
time_XBART <- round(time[3], 3)

################################
# predict on testing set
pred <- predict(fit, xtest)
pred <- rowMeans(pred[, params$burnin:params$num_sweeps])



# #####
# # bart with default initialization
fit_bart <- wbart(x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, ndpost = 200, nskip = 0)

# pred_bart = colMeans(predict(fit_bart, xtest))


# bart with XBART initialization
fit_bart2 <- wbart_ini(treedraws = fit$treedraws, x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, nskip = 0, ndpost = 100, sigest = mean(fit$sigma))


pred_bart_ini <- colMeans(predict(fit_bart2, xtest))


xbart_rmse <- sqrt(mean((fhat.1 - ftest)^2))
bart_rmse <- sqrt(mean((pred_bart - ftest)^2))
bart_ini_rmse <- sqrt(mean((pred_bart_ini - ftest)^2))


xbart_rmse
bart_rmse
bart_ini_rmse



#######################################################################
# Calculate coverage
#######################################################################

# coverage of the real average
draw_BART_XBART <- c()

for (i in 15:50) {
    # bart with XBART initialization
    cat("------------- i ", i, "\n")
    set.seed(1)
    fit_bart2 <- wbart_ini(treedraws = fit$treedraws[i], x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, nskip = 0, ndpost = 100)

    draw_BART_XBART <- rbind(draw_BART_XBART, fit_bart2$yhat.test)
}

i <- 20
set.seed(1)
fit_bart2 <- wbart_ini(treedraws = fit$treedraws[i], x, y, x.test = xtest, numcut = params$num_cutpoints, ntree = params$num_trees, nskip = 0, ndpost = 100)
plot(fit_bart2$yhat.test[1, ])


# #######################################################################
# # print
xbart_rmse <- sqrt(mean((fhat.1 - ftest)^2))
bart_rmse <- sqrt(mean((pred_bart - ftest)^2))
bart_ini_rmse <- sqrt(mean((colMeans(draw_BART_XBART) - ftest)^2))


xbart_rmse
bart_rmse
bart_ini_rmse



coverage <- c(0, 0, 0)

length <- matrix(0, nt, 3)

for (i in 1:nt) {
    lower <- quantile(fit$yhats_test[i, 15:50], 0.025)
    higher <- quantile(fit$yhats_test[i, 15:50], 0.975)
    if (ftest[i] < higher && ftest[i] > lower) {
        coverage[1] <- coverage[1] + 1
    }
    length[i, 1] <- higher - lower

    lower <- quantile(fit_bart$yhat.test[, i], 0.025)
    higher <- quantile(fit_bart$yhat.test[, i], 0.975)
    if (ftest[i] < higher && ftest[i] > lower) {
        coverage[2] <- coverage[2] + 1
    }
    length[i, 2] <- higher - lower

    lower <- quantile(draw_BART_XBART[, i], 0.025)
    higher <- quantile(draw_BART_XBART[, i], 0.975)
    if (ftest[i] < higher && ftest[i] > lower) {
        coverage[3] <- coverage[3] + 1
    }
    length[i, 3] <- higher - lower
}

coverage / nt
colMeans(length)