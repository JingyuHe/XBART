#' XBART main function of XBART regression.
#'
#' @param y A vector of outcome variable of length n, expected to be continuous.
#' @param X A matrix of input for the tree of size n by p. Column order matters: continuous features should all go before categorical. The number of categorical variables is p_categorical_con.
#' @param num_trees Integer, number of trees in the prognostic forest.
#' @param num_sweeps Integer, number of sweeps to fit for both forests.
#' @param max_depth Integer, maximum depth of the trees. The tree will stop grow if reaches the max depth.
#' @param Nmin Integer, minimal number of data points in a leaf. Any leaf will stop split if number of data points within is smaller than Nmin.
#' @param num_cutpoints Integer, number of cutpoint candidates to consider for each variable. Take in quantiles of the data.
#' @param alpha Scalar, BART prior parameter for trees. The default value is 0.95.
#' @param beta Scalar, BART prior parameter for trees. The default value is 1.25.
#' @param tau Scalar, prior parameter for the trees. The default value is 1 / num_trees.
#' @param no_split_penalty Extra weight of no-split option. The default value is 0, or you can take any other number greater than 0.
#' @param burnin Integer, number of burnin sweeps.
#' @param mtry Integer, number of X variables to sample at each split of the tree.
#' @param p_categorical Integer, number of categorical variables in X, note that all categorical variables should be put after continuous variables. Default value is 0.
#' @param kap Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 16.
#' @param s Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 4.
#' @param tau_kap Scalar, parameter of the inverse gamma prior on tau. Default value is 3.
#' @param tau_s Scalar, parameter of the inverse gamma prior on tau. Default value is 0.5.
#' @param verbose Bool, whether to print fitting process on the screen or not.
#' @param update_tau Bool, if TRUE, update the prior of leaf mean.
#' @param paralll Bool, whether to run in parallel on multiple CPU threads.
#' @param nthread Integer, number of threads to use if run in parallel.
#' @param random_seed Integer, random seed for replication.
#' @param sample_weights Bool, if TRUE, the weight to sample \eqn{X} variables at each tree will be sampled.
#'
#' @return A list contains fitted trees as well as parameter draws at each sweep.
#' @export



XBART <- function(y, X, num_trees, num_sweeps, max_depth = 250, Nmin = 1, num_cutpoints = 100, alpha = 0.95, beta = 1.25, tau = NULL, no_split_penalty = NULL, burnin = 1L, mtry = NULL, p_categorical = 0L, kap = 16, s = 4, tau_kap = 3, tau_s = 0.5, verbose = FALSE, update_tau = TRUE, parallel = TRUE, random_seed = NULL, sample_weights = TRUE, nthread = 0, ...) {
    if (!inherits(X, "matrix")) {
        warning("Input X is not a matrix, try to convert type.\n")
        X <- as.matrix(X)
    }

    if (!inherits(y, "matrix")) {
        warning("Input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    if (dim(X)[1] != length(y)) {
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

    if (is.null(tau)) {
        tau <- var(y) / num_trees
        cat("tau = var(y)/num_trees, default value. \n")
    }

    if (is.null(mtry)) {
        mtry <- dim(X)[2]
        cat("mtry = p, use all variables. \n")
    }

    if (mtry > dim(X)[2]) {
        mtry <- dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }

    if (p_categorical > dim(X)[2]) {
        p_categorical <- dim(X)[2]
        stop("p_categorical cannot exceed p")
    }
    # check input type

    check_non_negative_integer(burnin, "burnin")
    check_non_negative_integer(p_categorical, "p_categorical")

    check_positive_integer(max_depth, "max_depth")
    check_positive_integer(Nmin, "Nmin")
    check_positive_integer(num_sweeps, "num_sweeps")
    check_positive_integer(num_trees, "num_trees")
    check_positive_integer(num_cutpoints, "num_cutpoints")

    check_scalar(tau, "tau")
    check_scalar(no_split_penalty, "no_split_penalty")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")
    check_scalar(kap, "kap")
    check_scalar(s, "s")

    obj <- XBART_cpp(
        y, X, num_trees, num_sweeps, max_depth,
        Nmin, num_cutpoints, alpha, beta, tau, no_split_penalty, burnin,
        mtry, p_categorical, kap, s, tau_kap, tau_s, verbose, update_tau, parallel, set_random_seed,
        random_seed, sample_weights, nthread
    )

    # tree_json <- r_to_json(mean(y), obj$model$tree_pnt)
    # obj$tree_json <- tree_json

    class(obj) <- "XBART"
    return(obj)
}
