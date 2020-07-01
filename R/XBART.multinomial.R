XBART.multinomial <- function(y, num_class, X, Xtest, num_trees, num_sweeps, max_depth = 250, Nmin = 1, num_cutpoints = 100, alpha = 0.95, beta = 1.25, tau_a = NULL, tau_b = NULL, no_split_penality = NULL, burnin = 1L, mtry = NULL, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, random_seed = NULL, sample_weights_flag = TRUE, separate_tree = FALSE, stop_threshold = 0.0, nthread = 0, weight = 1, ...) {

    if (class(X) != "matrix") {
        cat("Input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }

    if (class(Xtest) != "matrix") {
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest = as.matrix(Xtest)
    }

    if (class(y) != "matrix") {
        cat("Input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    #TODO: Transform y back to original label after training?
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

    if (dim(X)[2] != dim(X)[2]) {
        stop("Column of X must match columns of Xtest")
    }

    if (is.null(random_seed)) {
        set_random_seed = FALSE
        random_seed = 0
    } else {
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed = TRUE
    }

    if (burnin >= num_sweeps) {
        stop("Burnin samples should be smaller than number of sweeps.\n")
    }

    if (is.null(no_split_penality) || no_split_penality == "Auto") {
        no_split_penality = log(num_cutpoints)
    }

    # if (is.null(tau)) {
    #     tau = 1 / num_trees
    #     cat("tau = 1/num_trees, default value. \n")
    # }

    if (is.null(mtry)) {
        mtry = dim(X)[2]
        cat("mtry = p, use all variables. \n")
    }

    if (mtry > dim(X)[2]) {
        mtry = dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }

    if (p_categorical > dim(X)[2]) {
        p_categorical = dim(X)[2]
        stop("p_categorical cannot exceed p")
    }

    if(is.null(weight)){
        weight = c(1)
    }

    if (length(weight) > 1) {
        cat("Input weight should be a single number, initialized as weight = ", weight[1], ".\n")
        weight = weight[1]
    }
    # check input type

    check_non_negative_integer(burnin, "burnin")
    check_non_negative_integer(p_categorical, "p_categorical")

    check_positive_integer(max_depth, "max_depth")
    check_positive_integer(Nmin, "Nmin")
    check_positive_integer(num_sweeps, "num_sweeps")
    check_positive_integer(num_trees, "num_trees")
    check_positive_integer(num_cutpoints, "num_cutpoints")

    # check_scalar(tau, "tau")
    check_scalar(no_split_penality, "no_split_penality")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")
    check_scalar(kap, "kap")
    check_scalar(s, "s")

    obj = XBART_multinomial_cpp(y, num_class, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau_a, tau_b, no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, set_random_seed, random_seed, sample_weights_flag, separate_tree, stop_threshold, nthread, weight)
    class(obj) = "XBARTmultinomial" # Change to XBARTProbit?
    return(obj)
}