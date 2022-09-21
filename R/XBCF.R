#' Run XBCF model on data
#'
#' @param y An array of outcome variables of length n (expected to be continuos).
#' @param z A binary array of treatment assignments of length n.
#' @param x_con An input matrix for the prognostic term of size n by p1. Column order matters: continuos features should all bgo before of categorical.
#' @param x_mod An input matrix for the treatment term of size n by p2 (default x_mod = x_con). Column order matters: continuos features should all go beforeof categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated inside wsbcf using nnet function.
#' @param num_sweeps The total number of sweeps over data (default is 60).
#' @param num_burnin The number of burn-in sweeps (default is 20).
#' @param pihat An array of propensity score estimates of length n (default is NULL). In the default case propensity scores are evaluated within the wsbcf function with nnet.
#' @param max_depth The maximum possible depth of a tree.
#' @param Nmin The minimum node size.
#' @param num_cutpoints The number of adaptive cutpoints considered at each split for continuous variables (default is 100).
#' @param pcat_con The number of categorical inputs in the prognostic term input matrix x_con.
#' @param pcat_mod The number of categorical inputs in the treatment term input matrix x_mod.
#' @param n_trees_con The number of trees in the prognostic forest (default is 30).
#' @param n_trees_mod The number of trees in the treatment forest (default is 10).
#' @param alpha_con Base parameter for tree prior on trees in prognostic forest (default is 0.95).
#' @param beta_con Power parameter for tree prior on trees in prognostic forest (default is 1.25).
#' @param tau_con Prior variance over the mean on on trees in prognostic forest (default is 0.6*var(y)/n_trees_con).
#' @param alpha_mod Base parameter for tree prior on trees in treatment forest (default is 0.25).
#' @param beta_mod Power parameter for tree prior on trees in treatment forest (default is 3).
#' @param tau_mod Prior variance over the mean on on trees in treatment forest (default is 0.1*var(y)/n_trees_mod).
#' @param burnin Number of burnin sweeps of forests for prediction.
#' @param mtry_con Number of X variables to draw for each split in the prognostic forest (default is 0, use all variables).
#' @param mtry_mod Number of X variables to draw for eac split in the treatment forest (default is 0, use all variables).
#' @param kap_con Prior parameter for the inverse Gamma prior on residual variance in prognostic forest (default is 16).
#' @param s_con Prior parameter for the inverse Gamma prior on residual variance in prognostic forest (default is 4).
#' @param kap_mod Prior parameter for the inverse Gamma prior on residual variance in treatment forest (default is 16).
#' @param s_mod Prior parameter for the inverse Gamma prior on residual variance in prognostic forest (default is 4).
#' @param pr_scale Bool, if True, use half-cauchy prior. Default is FALSE.
#' @param trt_scale Bool, if True, use half-normal prior. Default is FALSE.
#' @param verbose Bool flag for printing fitting process on the screen or not.
#' @param parallel Bool flag for fitting in parallel or not.
#' @param random_seed Seed for random number generator.
#' @param sample_weights Bool flag for sampling variable importance or not.
#' @param a_scaling Bool, if True, update a.
#' @param b_scaling Bool, if True, update b0 and b1.
#'
#' @return A fit file, which contains the draws from the model as well as parameter draws at each sweep.
#' @export

XBCF <- function(y, z, x_con, x_mod = x_con, pihat = NULL,
                          num_sweeps = 60, burnin = 20,
                          max_depth = 50, Nmin = 1L,
                          num_cutpoints = 100,
                          no_split_penality = "Auto", mtry_con = 0L, mtry_mod = 0L,
                          pcat_con = NULL,
                          pcat_mod = pcat_con,
                          n_trees_con = 30L,
                          alpha_con = 0.95, beta_con = 1.25, tau_con = NULL,
                          kap_con = 16, s_con = 4,
                          pr_scale = FALSE,
                          n_trees_mod = 10L,
                          alpha_mod = 0.25, beta_mod = 3, tau_mod = NULL,
                          kap_mod = 16, s_mod = 4,
                          trt_scale = FALSE,
                          verbose = FALSE, parallel = TRUE,
                          random_seed = NULL, sample_weights = TRUE,
                          a_scaling = TRUE, b_scaling = TRUE) {

    # index = order(z, decreasing=TRUE)

    # y = y[index]
    # X = matrix(c(x[,1][index],x[,2][index],x[,3][index]),nrow=length(x[,1]))
    # z = z[index]

    if (!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        x_con <- as.matrix(x_con)
    }
    if (!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        x_mod <- as.matrix(x_mod)
    }
    if (!("matrix" %in% class(z))) {
        cat("Msg: input z is not a matrix, try to convert type.\n")
        z <- as.matrix(z)
    }
    if (!("matrix" %in% class(y))) {
        cat("Msg: input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    # compute pihat if it wasn't provided with the call
    if (is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz <- nnet::nnet(z ~ ., data = x_con, size = 3, rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat <- fitz$fitted.values
    }
    if (!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat <- as.matrix(pihat)
    }

    x_con <- cbind(pihat, x_con)
    p_X <- ncol(x_con)
    p_Xt <- ncol(x_mod)


    if (nrow(x_con) != nrow(x_mod)) {
        stop("row number mismatch for the two input matrices")
    }
    if (nrow(x_con) != nrow(y)) {
        stop(paste0("row number mismatch between X (", nrow(x_con), ") and y (", nrow(y), ")"))
    }
    if (nrow(x_con) != nrow(z)) {
        stop(paste0("row number mismatch between X (", nrow(x_con), ") and z (", nrow(z), ")"))
    }

    # check if p_categorical was not provided
    if (is.null(pcat_con)) {
        stop("number of categorical variables pcat_con is not specified")
    }
    if (is.null(pcat_mod)) {
        stop("number of categorical variables pcat_mod is not specified")
    }

    # check if p_categorical exceeds the number of columns
    if (pcat_con > p_X) {
        stop("number of categorical variables (pcat_con) cannot exceed number of columns")
    }
    if (pcat_mod > p_Xt) {
        stop("number of categorical variables (pcat_mod) cannot exceed number of columns")
    }

    # check if p_categorical is negative
    if (pcat_con < 0 || pcat_mod < 0) {
        stop("number of categorical values can not be negative: check pcat_con and pcat_mod")
    }

    # check if mtry exceeds the number of columns
    if (mtry_con > p_X) {
        cat("Msg: mtry value cannot exceed number of columns; set to default.\n")
        mtry_con <- 0
    }
    if (mtry_mod > p_Xt) {
        cat("Msg: mtry value cannot exceed number of columns; set to default.\n")
        mtry_mod <- 0
    }

    # check if mtry is negative
    if (mtry_con < 0) {
        cat("Msg: mtry value cannot exceed number of columns; set to default.\n")
        mtry_con <- 0
    }
    if (mtry_mod < 0) {
        cat("Msg: mtry value cannot exceed number of columns; set to default.\n")
        mtry_mod <- 0
    }

    # set defaults for taus if it wasn't provided with the call
    if (is.null(tau_con)) {
        tau_con <- 0.6 * var(y) / n_trees_con
    }
    # set defaults for taus if it wasn't provided with the call
    if (is.null(tau_mod)) {
        tau_mod <- 0.1 * var(y) / n_trees_mod
    }

    meany <- mean(y)
    y <- y - meany
    sdy <- sd(y)

    if (sdy == 0) {
        stop("y is a constant variable; sdy = 0")
    } else {
        y <- y / sdy
    }

    # compute default values for taus if none provided
    if (is.null(tau_con)) {
        tau_con <- 0.6 * var(y) / n_trees_con
    }

    if (is.null(tau_mod)) {
        tau_mod <- 0.1 * var(y) / n_trees_mod
    }

    if (is.null(random_seed)) {
        set_random_seed <- FALSE
        random_seed <- 0
    } else {
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed <- TRUE
    }

    if (burnin >= num_sweeps) {
        stop(paste0("burnin (", burnin, ") cannot exceed or match the total number of sweeps (", sweeps, ")"))
    }
    if (no_split_penality == "Auto") {
        no_split_penality <- log(num_cutpoints)
    }

    print(y)

    obj <- XBCF_cpp(
        y, x_con, x_mod, z,
        num_sweeps, burnin,
        max_depth, Nmin,
        num_cutpoints,
        no_split_penality, mtry_con, mtry_mod,
        pcat_con,
        pcat_mod,
        n_trees_con,
        alpha_con, beta_con, tau_con,
        kap_con, s_con,
        pr_scale,
        n_trees_mod,
        alpha_mod, beta_mod, tau_mod,
        kap_mod, s_mod,
        trt_scale,
        verbose, parallel, set_random_seed,
        random_seed, sample_weights,
        a_scaling, b_scaling
    )
    class(obj) <- "XBCF"

    # obj$sdy_use = sdy_use
    obj$sdy <- sdy
    obj$meany <- meany
    obj$tauhats <- obj$tauhats * sdy
    obj$muhats <- obj$muhats * sdy

    obj$tauhats.adjusted <- matrix(NA, length(y), num_sweeps - burnin)
    obj$muhats.adjusted <- matrix(NA, length(y), num_sweeps - burnin)
    seq <- (burnin + 1):num_sweeps
    for (i in seq) {
        obj$tauhats.adjusted[, i - burnin] <- obj$tauhats[, i] * (obj$b_draws[i, 2] - obj$b_draws[i, 1])
        obj$muhats.adjusted[, i - burnin] <- obj$muhats[, i] * (obj$a_draws[i]) + meany
    }
    return(obj)
}
