XBART <- function(y, X, num_trees, num_sweeps, max_depth = 250, Nmin = 1, num_cutpoints = 100, alpha = 0.95, beta = 1.25, tau = NULL, no_split_penality = NULL, burnin = 1L, mtry = NULL, p_categorical = 0L, kap = 16, s = 4, tau_kap = 3, tau_s = 0.5, verbose = FALSE, sampling_tau = TRUE, parallel = TRUE, random_seed = NULL, sample_weights = TRUE, nthread = 0, ...) {
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

    if (is.null(no_split_penality) || no_split_penality == "Auto") {
        no_split_penality <- log(num_cutpoints)
    }

    if (is.null(tau)) {
        tau <- 1 / num_trees
        cat("tau = 1/num_trees, default value. \n")
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
    check_scalar(no_split_penality, "no_split_penality")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")
    check_scalar(kap, "kap")
    check_scalar(s, "s")

    obj <- XBART_cpp(
        y, X, num_trees, num_sweeps, max_depth,
        Nmin, num_cutpoints, alpha, beta, tau, no_split_penality, burnin,
        mtry, p_categorical, kap, s, tau_kap, tau_s, verbose, sampling_tau, parallel, set_random_seed,
        random_seed, sample_weights, nthread
    )

    tree_json <- r_to_json(mean(y), obj$model$tree_pnt)
    obj$tree_json <- tree_json

    class(obj) <- "XBART"
    return(obj)
}