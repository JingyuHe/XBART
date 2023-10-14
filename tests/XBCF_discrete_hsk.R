# simple demonstration of XBCF with default parameters
library(XBART)
library(dbarts)

reps <- 30
rmse.stats <- matrix(NA, nrow = reps, ncol = 8)
colnames(rmse.stats) <- c("tau, model3", "mu, model3", "tau, model2", "mu, model2", "tau, hsk", "mu, hsk", "tau, XBCF", "mu, XBCF")

a_scaling <- TRUE
b_scaling <- TRUE

set.seed(100)

for (i in c(1:reps)) {
    #### 1. DATA GENERATION PROCESS
    n <- 5000 # number of observations
    # set seed here
    # set.seed(1)

    # generate dcovariates
    x1 <- rnorm(n)
    x2 <- rbinom(n, 1, 0.2)
    x3 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
    x4 <- rnorm(n)
    x5 <- rbinom(n, 1, 0.7)
    x <- cbind(x1, x2, x3, x4, x5)

    # define treatment effects
    tau <- 3 + 0.5 * x[, 4] * (2 * x[, 5] - 1)

    ## define prognostic function (RIC)
    mu <- function(x) {
        lev <- c(-0.5, 0.75, 0)
        result <- 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x3]
        return(result)
    }

    # compute propensity scores and treatment assignment
    pi <- pnorm(-0.5 + mu(x) - x[, 2] + 0. * x[, 4], 0, 3)
    # hist(pi,100)
    z <- rbinom(n, 1, pi)

    # generate outcome variable
    # Ey <- mu(x) + tau * z
    Ey <- mu(x) + tau * z
    # sig <- .8 * exp(x[, 1])
    sig <- .8 * exp(x[, 1])# + z * exp(x[, 2]) #+ 0.25 * sd(Ey) # exponential function s
    y <- Ey + 0.5 * sig * rnorm(n)

    # If you didn't know pi, you would estimate it here
    pihat <- pi

    # store mu values before processing input matrix
    muvec <- mu(x)

    # matrix prep
    x <- data.frame(x)
    x[, 3] <- as.factor(x[, 3])
    x <- makeModelMatrixFromDataFrame(data.frame(x))
    x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])

    # add pihat to the prognostic term matrix
    x1 <- cbind(pihat, x)

    x_con <- x
    x_mod <- x


    #### 2. XBCF, heterosk, treatment modified variance

    num_sweeps <- 100
    burnin <- 40
    # run XBCF heteroskedastic
    t1 <- proc.time()
    fit.hsk <- XBCF.discrete.heterosk3(
        y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat,
        p_categorical_con = 5, p_categorical_mod = 5,
        num_trees_con = 5, num_trees_mod = 5,
        num_sweeps = num_sweeps, burnin = burnin, sample_weights = TRUE,
        a_scaling = a_scaling, b_scaling = b_scaling
    )
    t1 <- proc.time() - t1
    cat(t1, "\n")

    pred <- predict.XBCFdiscreteHeterosk3(fit.hsk, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
    tauhats <- pred$tau.adj.mean
    muhats <- pred$mu.adj.mean

    sigma <- sqrt(rowMeans(pred$variance[, burnin:num_sweeps]))
    sigma_con <- sqrt(rowMeans(pred$variance_con[, burnin:num_sweeps]))
    # compare results to inference
    # plot(tau, tauhats)
    # abline(0, 1)
    # plot(mu(x), muhats)
    # abline(0, 1)
    cat("++++++++++++++++++++++++++++++++\n")
    print(paste0("xbcf-het tau RMSE: ", sqrt(mean((tauhats - tau)^2))))
    print(paste0("xbcf-het mu RMSE: ", sqrt(mean((muhats - muvec)^2))))
    print(paste0("xbcf-het runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))

    rmse.stats[i, 1] <- sqrt(mean((tauhats - tau)^2))
    rmse.stats[i, 2] <- sqrt(mean((muhats - muvec)^2))


    # XBCF, heterosk, separate variance forest for treated / control


    num_sweeps <- 60
    burnin <- 30

    t1 <- proc.time()
    fit.hsk4 <- XBCF.discrete.heterosk2(
        y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat,
        p_categorical_con = 5, p_categorical_mod = 5,
        num_trees_con = 5, num_trees_mod = 5,
        num_sweeps = num_sweeps, burnin = burnin,
        a_scaling = a_scaling, b_scaling = b_scaling
    )
    t1 <- proc.time() - t1
    cat(t1, "\n")

    pred4 <- predict.XBCFdiscreteHeterosk2(fit.hsk4, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
    tauhats4 <- pred4$tau.adj.mean
    muhats4 <- pred4$mu.adj.mean


    rmse.stats[i, 3] <- sqrt(mean((tauhats4 - tau)^2))
    rmse.stats[i, 4] <- sqrt(mean((muhats4 - muvec)^2))

    # XBCF, heterosk, single variance forest

    num_sweeps <- 60
    burnin <- 30
    # run XBCF heteroskedastic
    t1 <- proc.time()
    fit.hsk2 <- XBCF.discrete.heterosk(
        y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat,
        p_categorical_con = 5, p_categorical_mod = 5,
        num_trees_con = 5, num_trees_mod = 5,
        
        num_sweeps = num_sweeps, burnin = burnin,
        a_scaling = a_scaling, b_scaling = b_scaling
    )
    t1 <- proc.time() - t1
    cat(t1, "\n")

    pred2 <- predict.XBCFdiscreteHeterosk(fit.hsk2, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
    tauhats2 <- pred2$tau.adj.mean
    muhats2 <- pred2$mu.adj.mean


    rmse.stats[i, 5] <- sqrt(mean((tauhats2 - tau)^2))
    rmse.stats[i, 6] <- sqrt(mean((muhats2 - muvec)^2))


    # XBCF, homoskedastic


    num_sweeps <- 60
    burnin <- 30
    
    t2 <- proc.time()
    fit <- XBCF.discrete(
        y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat,
        p_categorical_con = 5, p_categorical_mod = 5,
        num_sweeps = num_sweeps, burnin = burnin,
        num_trees_con = 5, num_trees_mod = 5,
        a_scaling = a_scaling, b_scaling = b_scaling
    )
    t2 <- proc.time() - t2

    pred3 <- predict(fit, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat, burnin = burnin)
    tauhats3 <- pred3$tau.adj.mean
    muhats3 <- pred3$mu.adj.mean

    rmse.stats[i, 7] <- sqrt(mean((tauhats3 - tau)^2))
    rmse.stats[i, 8] <- sqrt(mean((muhats3 - muvec)^2))

    cat("------------------------------\n")
    print(paste0("xbcf tau RMSE: ", sqrt(mean((tauhats3 - tau)^2))))
    print(paste0("xbcf tau RMSE: ", sqrt(mean((muhats3 - muvec)^2))))
    print(paste0("xbcf runtime: ", round(as.list(t2)$elapsed, 2), " seconds"))

    # check predicted outcomes
    # plot(y,rowMeans(pred$yhats.adj[,30:60]))
    par(mfrow = c(2, 4))
    plot(tau, tauhats, main = "tau, model3")
    abline(0, 1)
    plot(tau, tauhats4, main = "tau, model2")
    abline(0, 1)
    plot(tau, tauhats2, main = "tau, hsk")
    abline(0, 1)
    plot(tau, tauhats3, main = "tau, XBCF")
    abline(0, 1)
    plot(muvec, muhats, main = "mu, model3")
    abline(0, 1)
    plot(muvec, muhats4, main = "mu, model2")
    abline(0, 1)
    plot(muvec, muhats2, main = "mu, hsk")
    abline(0, 1)
    plot(muvec, muhats3, main = "mu, XBCF")
    abline(0, 1)
    # plot(sig * rep(1, n), sigma, main = "sigma, hsk")
    # abline(0, 1)
}

cat(paste("Average RMSE for all simulations.\n"))
print(round(colMeans(rmse.stats), 3))

# main model parameters can be retrieved below
# print(xbcf.fit$model_params)
