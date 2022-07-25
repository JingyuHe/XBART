XBCF_continuous <- function(y, Z, X, X_ps, X_trt, Xtest, Xtest_ps, Xtest_trt, Ztest, num_trees, num_trees_ps, num_trees_trt, num_sweeps, max_depth = 250,
    Nmin = 1, num_cutpoints = 100, alpha = 0.95, beta = 1.25, tau = NULL,
    no_split_penality = NULL, burnin = 1L, mtry = NULL, mtry_ps = NULL, mtry_trt = NULL, p_categorical = 0L, p_categorical_ps = 0L, p_categorical_trt = 0L,
    kap = 16, s = 4, tau_kap = 3, tau_s = 0.5, verbose = FALSE, sampling_tau = TRUE, parallel = TRUE, random_seed = NULL,
    sample_weights_flag = TRUE, nthread = 0, ...) {

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

    if (!("matrix" %in% class(Z))) {
        cat("Input Z is not a matrix, try to convert type.\n")
        Z = as.matrix(Z)
    }

    if (dim(Z)[1] != length(y)) {
        stop("Length of Z must match length of y")
    }

    if (dim(X)[1] != length(y)) {
        stop("Length of X must match length of y")
    }

    if (dim(Ztest)[1] != dim(Xtest)[1]) {
        stop("Length of Ztest must match length of Xtest")
    }

    if (dim(X)[2] != dim(Xtest)[2]) {
        stop("Column of X must match columns of Xtest")
    }

    if (dim(Z)[2] != dim(Ztest)[2]) {
        stop("Column of Zt must match length of Ztest")
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

    if (is.null(tau)) {
        tau = 1/num_trees
        cat("tau = 1/num_trees, default value. \n")
    }

    if (is.null(mtry)) {
        mtry = dim(X)[2]
        cat("mtry = p, use all variables. \n")
    }

    if (mtry > dim(X)[2]){
        mtry = dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }    
    
    if (is.null(mtry_ps)) {
        mtry_ps = dim(X_ps)[2]
        cat("mtry_ps = p_ps, use all variables. \n")
    }

    if (mtry_ps > dim(X_ps)[2]){
        mtry_ps = dim(X_ps)[2]
        cat("mtry_ps cannot exceed p_ps, set to mtry_ps = p_ps. \n")
    }    
    
    if (is.null(mtry_trt)) {
        mtry_trt = dim(X_trt)[2]
        cat("mtry_trt = p_trt, use all variables. \n")
    }

    if (mtry_trt > dim(X_trt)[2]){
        mtry_trt = dim(X_trt)[2]
        cat("mtry_trt cannot exceed p_trt, set to mtry_trt = p_trt. \n")
    }

    if(p_categorical > dim(X)[2]){
        p_categorical = dim(X)[2]
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

    obj = XBCF_continuous_cpp(y, Z, X, X_ps, X_trt, Xtest, Xtest_ps, Xtest_trt, Ztest, num_trees, num_trees_ps, num_trees_trt, num_sweeps, max_depth,
        Nmin, num_cutpoints, alpha, beta, tau, no_split_penality, burnin,
        mtry, mtry_ps, mtry_trt, p_categorical, p_categorical_ps, p_categorical_trt, kap, s, tau_kap, tau_s, verbose, sampling_tau, parallel, set_random_seed,
        random_seed, sample_weights_flag, nthread)

    #tree_json = r_to_json(mean(y), obj$model$tree_pnt)
    #obj$tree_json = tree_json

    class(obj) = "XBART"
    return(obj)
}
