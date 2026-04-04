###################################################
# Regression reproducibility check for XBART
# This script verifies that passing random_seed to
# XBART fixes the full stochastic path, including
# tree draws and predictions.
###################################################

get_script_path <- function() {
    file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    normalizePath(sub("^--file=", "", file_arg[1]))
}


load_xbart_package <- function() {
    repo_root <- normalizePath(file.path(dirname(get_script_path()), ".."))

    if (requireNamespace("XBART", quietly = TRUE)) {
        suppressPackageStartupMessages(library(XBART))
    } else {
        if (!requireNamespace("pkgload", quietly = TRUE)) {
            stop("XBART is not installed and pkgload is unavailable.")
        }
        pkgload::load_all(repo_root, export_all = FALSE, quiet = TRUE)
    }
}


load_xbart_package()


simulate_regression_data <- function(data_seed = 20260404L,
                                     n = 300L,
                                     nt = 120L,
                                     d = 8L,
                                     dcat = 2L) {
    stopifnot(d > dcat)

    set.seed(data_seed)

    x_cont <- matrix(runif((d - dcat) * n, -2, 2), n, d - dcat)
    x_cat <- matrix(sample(-2:2, dcat * n, replace = TRUE), n, dcat)
    x <- cbind(x_cont, x_cat)

    xtest_cont <- matrix(runif((d - dcat) * nt, -2, 2), nt, d - dcat)
    xtest_cat <- matrix(sample(-2:2, dcat * nt, replace = TRUE), nt, dcat)
    xtest <- cbind(xtest_cont, xtest_cat)

    f <- function(z) {
        sin(z[, 1]) + z[, 2]^2 - z[, 3] * z[, 4] + 0.5 * z[, 5] * z[, 6]
    }

    mu <- f(x)
    mu_test <- f(xtest)
    y <- mu + sd(mu) * rnorm(n)

    list(
        x = x,
        xtest = xtest,
        y = y,
        mu_test = mu_test,
        p_categorical = dcat
    )
}


fit_xbart_once <- function(dat,
                           model_seed = 123L,
                           parallel_flag = FALSE,
                           nthread = if (parallel_flag) 2L else 1L) {
    fit <- XBART(
        y = as.matrix(dat$y),
        X = as.matrix(dat$x),
        num_trees = 8,
        num_sweeps = 6,
        max_depth = 50,
        Nmin = 1,
        num_cutpoints = 16,
        burnin = 2,
        mtry = 4,
        p_categorical = dat$p_categorical,
        tau = var(dat$y) / 8,
        kap = 1,
        s = 1,
        verbose = FALSE,
        parallel = parallel_flag,
        random_seed = model_seed,
        sample_weights = TRUE,
        nthread = nthread
    )

    pred <- predict(fit, dat$xtest)

    list(
        fit = fit,
        pred = pred,
        rmse = sqrt(mean((rowMeans(pred[, 3:6]) - dat$mu_test)^2))
    )
}


assert_same_seed_reproducible <- function(dat, parallel_flag) {
    fit1 <- fit_xbart_once(dat, model_seed = 123L, parallel_flag = parallel_flag)
    fit2 <- fit_xbart_once(dat, model_seed = 123L, parallel_flag = parallel_flag)
    fit3 <- fit_xbart_once(dat, model_seed = 124L, parallel_flag = parallel_flag)

    stopifnot(identical(fit1$pred, fit2$pred))
    stopifnot(identical(fit1$fit$sigma, fit2$fit$sigma))
    stopifnot(identical(fit1$fit$importance, fit2$fit$importance))
    stopifnot(identical(fit1$fit$treedraws, fit2$fit$treedraws))
    stopifnot(identical(fit1$fit$tree_json[[1]], fit2$fit$tree_json[[1]]))
    stopifnot(!identical(fit1$pred, fit3$pred))

    cat("parallel =", parallel_flag, "\n")
    cat("same-seed prediction identical: TRUE\n")
    cat("same-seed tree draws identical: TRUE\n")
    cat("different-seed prediction differs: TRUE\n")
    cat("rmse:", round(fit1$rmse, 4), "\n")
    cat("----\n")
}


check_r_seed_only <- function(dat) {
    run_with_r_seed_only <- function(r_seed) {
        set.seed(r_seed)
        fit <- XBART(
            y = as.matrix(dat$y),
            X = as.matrix(dat$x),
            num_trees = 8,
            num_sweeps = 6,
            max_depth = 50,
            Nmin = 1,
            num_cutpoints = 16,
            burnin = 2,
            mtry = 4,
            p_categorical = dat$p_categorical,
            tau = var(dat$y) / 8,
            kap = 1,
            s = 1,
            verbose = FALSE,
            parallel = FALSE,
            sample_weights = TRUE,
            nthread = 1
        )

        predict(fit, dat$xtest)
    }

    pred1 <- run_with_r_seed_only(999L)
    pred2 <- run_with_r_seed_only(999L)

    identical_r_seed_only <- identical(pred1, pred2)

    cat("R set.seed() only identical without random_seed:",
        identical_r_seed_only, "\n")

    if (!identical_r_seed_only) {
        cat("Note: reproducibility currently requires passing random_seed to XBART().\n")
    }
}


main <- function() {
    dat <- simulate_regression_data()

    assert_same_seed_reproducible(dat, parallel_flag = FALSE)
    assert_same_seed_reproducible(dat, parallel_flag = TRUE)
    check_r_seed_only(dat)

    cat("All regression reproducibility checks passed.\n")
}


main()
