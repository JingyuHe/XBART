#' XBART main function of XBART classification.
#'
#' @param y A vector of outcome variable of length n, expected to be discrete.
#' @param num_class Integer, number of different unique classes for the classification task.
#' @param X A matrix of input for the tree of size n by p. Column order matters: continuous features should all go before categorical. The number of categorical variables is p_categorical_con.
#' @param num_trees Integer, number of trees in the prognostic forest.
#' @param num_sweeps Integer, number of sweeps to fit for both forests.
#' @param max_depth Integer, maximum depth of the trees. The tree will stop grow if reaches the max depth.
#' @param Nmin Integer, minimal number of data points in a leaf. Any leaf will stop split if number of data points within is smaller than Nmin.
#' @param num_cutpoints Integer, number of cutpoint candidates to consider for each variable. Take in quantiles of the data.
#' @param alpha Scalar, BART prior parameter for trees. The default value is 0.95.
#' @param beta Scalar, BART prior parameter for trees. The default value is 1.25.
#' @param tau_a Scalar, prior of the leaf mean.
#' @param tau_b Scalar, prior of the leaf mean.
#' @param no_split_penalty Extra weight of no-split option. The default value is 1, or you can take any other number greater than 0.
#' @param burnin Integer, number of burnin sweeps.
#' @param mtry Integer, number of X variables to sample at each split of the tree.
#' @param p_categorical Integer, number of categorical variables in X, note that all categorical variables should be put after continuous variables. Default value is 0.
#' @param kap Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 16.
#' @param s Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 4.
#' @param tau_kap Scalar, parameter of the inverse gamma prior on tau. Default value is 3.
#' @param tau_s Scalar, parameter of the inverse gamma prior on tau. Default value is 0.5.
#' @param verbose Bool, whether to print fitting process on the screen or not.
#' @param paralll Bool, whether to run in parallel on multiple CPU threads.
#' @param nthread Integer, number of threads to use if run in parallel.
#' @param random_seed Integer, random seed for replication.
#' @param sample_weights Bool, if TRUE, the weight to sample \eqn{X} variables at each tree will be sampled.
#' @param separate_tree Bool, if TRUE, fit separate trees for different classes, otherwise all classes share the same tree strucutre.
#' @param weight Replicate factor of the Poisson observations. The default value is 1.
#' @param update_weight Bool, if TRUE, sample the replicate factor to reflect the data entropy.
#' @param update_tau Bool, if TRUE, update the prior of leaf mean.
#' @param hmult Prior of the replicate factor.
#' @param heps Prior of the replicate factor
#' @param ... optional parameters to be passed to the low level function XBART
#'
#' @details XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (with- out burnin sweeps). This function fits trees for multinomial classification tasks. Note that users have option to fit different tree structure for different classes, or let all classes share the same tree structure.
#' @return A list contains fitted trees as well as parameter draws at each sweep.
#' @export



XBART.multinomial <- function(y, num_class, X, num_trees = 20, num_sweeps = 20, max_depth = 20, Nmin = NULL, num_cutpoints = NULL, alpha = 0.95, beta = 1.25, tau_a = 1, tau_b = 1, no_split_penalty = NULL, burnin = 5, mtry = NULL, p_categorical = 0L, verbose = FALSE, parallel = TRUE, random_seed = NULL, sample_weights = TRUE, separate_tree = FALSE, weight = 1, update_weight = TRUE, update_tau = TRUE, nthread = 0, hmult = 1, heps = 0.1, ...) {
    require(GIGrvg)
    if (!("matrix" %in% class(X))) {
        cat("Input X is not a matrix, try to convert type.\n")
        X <- as.matrix(X)
    }

    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    if (is.null(Nmin)) {
        # cat("Nmin = ", 3 * num_class, " by default, \n")
        Nmin <- 3 * num_class
    }

    if (is.null(num_cutpoints)) {
        # cat("num_cutpoints = ", 20, " by default. \n")
        num_cutpoints <- 20
    }

    # TODO: Transform y back to original label after training?
    # if (class(as.vector(y)) != "numeric"){
    #     cat("Transform y to numeric label.\n")
    #     y = as.numeric( as.factor(y)) - 1
    #     y = as.matrix(y)
    # }else if(any (sort(unique(y)) != 0:(num_class -1)))
    # {
    #     cat("Transform y to numeric label start with 0.\n")
    #     y = as.numeric( as.factor(y)) - 1
    #     y = as.matrix(y)
    # }


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

    if (is.null(mtry)) {
        p <- dim(X)[2]
        if (p <= 10) {
            mtry <- p
        } else if (p <= 100) {
            mtry <- ceiling(p / 2)
        } else {
            mtry <- ceiling(p / 3)
        }
        cat("mtry = ", mtry, " by default. \n")
    }

    if (mtry > dim(X)[2]) {
        mtry <- dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }

    if (p_categorical > dim(X)[2]) {
        p_categorical <- dim(X)[2]
        stop("p_categorical cannot exceed p")
    }

    if (is.null(weight)) {
        weight <- 1
    }

    # check input type
    check_non_negative_integer(burnin, "burnin")
    check_non_negative_integer(p_categorical, "p_categorical")
    check_non_negative_integer(nthread, "nthread")

    check_positive_integer(max_depth, "max_depth")
    check_positive_integer(Nmin, "Nmin")
    check_positive_integer(num_sweeps, "num_sweeps")
    check_positive_integer(num_trees, "num_trees")
    check_positive_integer(num_cutpoints, "num_cutpoints")
    check_positive_integer(mtry, "mtry")

    # check_scalar(tau, "tau")
    check_scalar(no_split_penalty, "no_split_penalty")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")

    obj <- XBART_multinomial_cpp(y, num_class, X, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau_a, tau_b, no_split_penalty, burnin, mtry, p_categorical, verbose, parallel, set_random_seed, random_seed, sample_weights, separate_tree, weight, update_weight, update_tau, nthread, hmult, heps)

    class(obj) <- "XBARTmultinomial"

    if (separate_tree) {
        tree_json <- r_to_json_3D(obj$tree_pnt)
        obj$tree_json <- tree_json
        obj$separate_tree <- separate_tree
    } else {
        tree_json <- r_to_json(0, obj$tree_pnt)
        obj$tree_json <- tree_json
        obj$separate_tree <- separate_tree
    }
    return(obj)
}
