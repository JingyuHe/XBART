#' XBCF causal forest for continuous treatment variable.
#' @description This function fits XBCF causal forest model with continuous treatment variable. In sepecific, the model is \eqn{y = \tau(X_{con}) + \mu(X_{mod}) \times z + \epsilon, \quad \epsilon\sim N(0, \sigma^2)}. Where the \eqn{\tau(X_{con})} and \eqn{\mu(X_{mod})} are two XBART forest called prognostic and treatment forest respectively.
#' @param y A vector of outcome variable of length n, expected to be continuous.
#' @param Z A vector of treatment variable of length n, expected to be continuous.
#' @param X_con A matrix of input for the prognostic forest of size n by p_con. Column order matters: continuous features should all go before categorical. The number of categorical variables is p_categorical_con.
#' @param X_mod A matrix of input for the treatment forest of size n by p_mod. Column order matters: continuous features should all go before categorical. The number of categorical variables is p_categorical_mod.
#' @param num_trees_con Integer, number of trees in the prognostic forest.
#' @param num_trees_mod Integer, number of trees in the treatment forest.
#' @param num_sweeps Integer, number of sweeps to fit for both forests.
#' @param max_depth Integer, maximum depth of the trees. The tree will stop grow if reaches the max depth.
#' @param Nmin Integer, minimal number of data points in a leaf. Any leaf will stop split if number of data points within is smaller than Nmin.
#' @param num_cutpoints Integer, number of cutpoint candidates to consider for each variable. Take in quantiles of the data.
#' @param alpha_con Scalar, BART prior parameter for prognostic forest. The default value is 0.95.
#' @param beta_con Scalar, BART prior parameter for prognostic forest. The default value is 1.25.
#' @param alpha_mod Scalar, BART prior parameter for treatment forest. The default value is 0.25.
#' @param beta_mod Scalar, BART prior parameter for treatment forest. The default value is 3.
#' @param tau_con Scalar, prior parameter for prognostic forest. The default value is 0.6 * var(y) / num_trees_con.
#' @param tau_mod Scalar, prior parameter for treatment forest. The default value is 0.1 * var(y) / num_trees_mod.
#' @param no_split_penalty Weight of no-split option. The default value is log(num_cutpoints), or you can take any other number in log scale.
#' @param burnin Integer, number of burnin sweeps.
#' @param mtry_con Integer, number of X variables to sample at each split of the prognostic forest.
#' @param mtry_mod Integer, number of X variables to sample at each split of the treatment forest.
#' @param p_categorical_con Integer, number of categorical variables in X_con, note that all categorical variables should be put after continuous variables. Default value is 0.
#' @param p_categorical_mod Integer, number of categorical variables in X_mod, note that all categorical variables should be put after continuous variables. Default value is 0.
#' @param kap Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 16.
#' @param s Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 4.
#' @param tau_con_kap Scalar, parameter of the inverse gamma prior on tau_con. Default value is 3.
#' @param tau_con_s Scalar, parameter of the inverse gamma prior on tau_con. Default value is 0.5.
#' @param tau_mod_kap Scalar, parameter of the inverse gamma prior on tau_mod. Default value is 3.
#' @param tau_mod_s Scalar, parameter of the inverse gamma prior on tau_mod. Default value is 0.5.
#' @param verbose Bool, whether to print fitting process on the screen or not.
#' @param update_tau Bool. If TRUE, update the prior of leaf mean.
#' @param paralll Bool, whether to run in parallel on multiple CPU threads.
#' @param nthread Integer, number of threads to use if run in parallel.
#' @param random_seed Integer, random seed for replication.
#' @param sample_weights Bool, if TRUE, the weight to sample \eqn{X} variables at each tree will be sampled.
#'
#' @return A list contains fitted trees as well as parameter draws at each sweep.
#' @export


XBCF.continuous <- function(y, Z, X_con, X_mod, num_trees_con, num_trees_mod, num_sweeps, max_depth = 250, Nmin = 1, num_cutpoints = 100, alpha_con = 0.95, beta_con = 1.25, alpha_mod = 0.25, beta_mod = 3, tau_con = NULL, tau_mod = NULL, no_split_penalty = NULL, burnin = 1L, mtry_con = NULL, mtry_mod = NULL, p_categorical_con = 0L, p_categorical_mod = 0L, kap = 16, s = 4, tau_con_kap = 3, tau_con_s = 0.5, tau_mod_kap = 3, tau_mod_s = 0.5, verbose = FALSE, update_tau = TRUE, parallel = TRUE, random_seed = NULL, sample_weights = TRUE, nthread = 0, ...) {
    if (!("matrix" %in% class(X_con))) {
        cat("Input X_con is not a matrix, try to convert type.\n")
        X_con <- as.matrix(X_con)
    }

    if (!("matrix" %in% class(X_mod))) {
        cat("Input X_mod is not a matrix, try to convert type.\n")
        X_mod <- as.matrix(X_mod)
    }

    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    if (!("matrix" %in% class(Z))) {
        cat("Input Z is not a matrix, try to convert type.\n")
        Z <- as.matrix(Z)
    }

    if (dim(Z)[1] != length(y)) {
        stop("Length of Z must match length of y")
    }

    if (dim(X_con)[1] != length(y)) {
        stop("Length of X must match length of y")
    }

    if (is.null(random_seed)) {
        set_random_seed <- FALSE
        random_seed <- 0
    } else {
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed <- TRUE
    }

    if (burnin >= num_sweeps) {
        stop("Burnin samples should be smaller than number of sweeps.\n")
    }

    if (is.null(no_split_penalty) || no_split_penalty == "Auto") {
        no_split_penalty <- log(1)
    } else {
        no_split_penalty <- log(no_split_penalty)
    }

    if (is.null(tau_con)) {
        tau_con <- 0.6 * var(y) / num_trees_con
        cat("tau_con = 0.6*var(y)/num_trees_con, default value. \n")
    }

    if (is.null(tau_mod)) {
        tau_mod <- 0.1 * var(y) / num_trees_mod
        cat("tau_mod = 0.1*var(y)/num_trees_mod, default value. \n")
    }

    if (is.null(mtry_con)) {
        mtry_con <- dim(X_con)[2]
        cat("mtry_con = p_con, use all variables. \n")
    }

    if (mtry_con > dim(X_con)[2]) {
        mtry_con <- dim(X_con)[2]
        cat("mtry_con cannot exceed p_con, set to mtry_con = p_con. \n")
    }

    if (is.null(mtry_mod)) {
        mtry_mod <- dim(X_mod)[2]
        cat("mtry_mod = p_mod, use all variables. \n")
    }

    if (mtry_mod > dim(X_mod)[2]) {
        mtry_mod <- dim(X_mod)[2]
        cat("mtry_mod cannot exceed p_mod, set to mtry_mod = p_mod. \n")
    }

    if (p_categorical_con > dim(X_con)[2]) {
        p_categorical_con <- dim(X_con)[2]
        stop("p_categorical cannot exceed p")
    }
    # check input type

    check_non_negative_integer(burnin, "burnin")

    check_positive_integer(max_depth, "max_depth")
    check_positive_integer(Nmin, "Nmin")
    check_positive_integer(num_sweeps, "num_sweeps")
    check_positive_integer(num_cutpoints, "num_cutpoints")
    check_positive_integer(num_trees_con, "num_trees_con")
    check_positive_integer(num_trees_mod, "num_trees_mod")

    check_scalar(tau_con, "tau_con")
    check_scalar(tau_mod, "tau_mod")
    check_scalar(no_split_penalty, "no_split_penalty")
    check_scalar(alpha_con, "alpha_con")
    check_scalar(beta_con, "beta_con")
    check_scalar(alpha_mod, "alpha_mod")
    check_scalar(beta_mod, "beta_mod")
    check_scalar(kap, "kap")
    check_scalar(s, "s")

    obj <- XBCF_continuous_cpp(y, Z, X_con, X_mod, num_trees_con, num_trees_mod, num_sweeps, max_depth, Nmin, num_cutpoints, alpha_con, beta_con, alpha_mod, beta_mod, tau_con, tau_mod, no_split_penalty, burnin, mtry_con, mtry_mod, p_categorical_con, p_categorical_mod, kap, s, tau_con_kap, tau_con_s, tau_mod_kap, tau_mod_s, verbose, update_tau, parallel, set_random_seed, random_seed, sample_weights, nthread)

    class(obj) <- "XBCFcontinuous"
    return(obj)
}
