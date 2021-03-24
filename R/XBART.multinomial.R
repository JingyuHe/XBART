XBART.multinomial <- function(y, num_class, X, Xtest, num_trees = 20, num_sweeps = 20, max_depth = 20, 
Nmin = NULL, num_cutpoints = NULL, alpha = 0.95, beta = 1.25, tau_a = 1, tau_b = 1, 
no_split_penality = NULL, burnin = 5, mtry = NULL, p_categorical = 0L, verbose = FALSE, 
parallel = TRUE, random_seed = NULL, sample_weights_flag = TRUE, separate_tree = FALSE, 
weight = 1, update_weight = TRUE, update_tau = TRUE, nthread = 0, hmult = 1, heps = 0.1, ...) {

    if (!("matrix" %in% class(X))) {
        cat("Input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }

    if (!("matrix" %in% class(Xtest))) {
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest = as.matrix(Xtest)
    }

    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    if (is.null(Nmin)) {
        # cat("Nmin = ", 3 * num_class, " by default, \n")
        Nmin = 3 * num_class
    }

    if (is.null(num_cutpoints)){
        # cat("num_cutpoints = ", 20, " by default. \n")
        num_cutpoints = 20
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
        p = dim(X)[2]
        if (p <= 10){mtry = p}
        else if (p <= 100) {mtry = ceiling(p/2)}
        else {mtry = ceiling(p/3)}
        cat("mtry = ", mtry, " by default. \n")
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
        weight = 1
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
    check_scalar(no_split_penality, "no_split_penality")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")

    obj = XBART_multinomial_cpp(y, num_class, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau_a, tau_b, 
    no_split_penality, burnin, mtry, p_categorical, verbose, parallel, set_random_seed, random_seed, sample_weights_flag, separate_tree, 
    weight, update_weight, update_tau, nthread, hmult, heps)
    class(obj) = "XBARTmultinomial"


    if(separate_tree){
        tree_json = r_to_json_3D(obj$tree_pnt)
        obj$tree_json = tree_json
        obj$separate_tree = separate_tree
    }else{
        tree_json = r_to_json(0, obj$tree_pnt)
        obj$tree_json = tree_json
        obj$separate_tree = separate_tree
    }
    return(obj)
}