# TODO: update the inputs for the dunction
XBART.heterosk <- function(y,
                           X,
                           Xtest,
                           num_sweeps,
                           burnin = 1L,
                           p_categorical = 0L,
                           mtry = NULL,
                           no_split_penality_m = NULL,
                           num_trees_m = 20,
                           max_depth_m = 250,
                           Nmin_m = 1,
                           num_cutpoints_m = 20,
                           tau_m = NULL,
                           no_split_penality_v = NULL,
                           num_trees_v = 5,
                           max_depth_v = 10,
                           Nmin_v = 50,
                           num_cutpoints_v = 100,
                           ini_var = 1, # optional initialization for variance
                           a_v = 1.0, b_v = 1.0,
                           alpha = 0.95,
                           beta = 1.25,
                           kap = 16, s = 4,
                           tau_kap = 3, tau_s = 0.5,
                           verbose = FALSE,
                           sampling_tau = TRUE,
                           parallel = TRUE,
                           random_seed = NULL,
                           sample_weights_flag = TRUE,
                           nthread = 0,
                           ...) {

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

    if (is.null(no_split_penality_m) || no_split_penality_m == "Auto") {
        no_split_penality_m <- log(1)
    } else {
        no_split_penality_m <- log(no_split_penality_m)
    }

    if (is.null(no_split_penality_v) || no_split_penality_v == "Auto") {
        no_split_penality_v <- log(1)
    } else {
        no_split_penality_v <- log(no_split_penality_v)
    }

    if (is.null(tau_m)) {
        tau_m = 1/num_trees_m
        cat("tau_m = 1/num_trees_m, default value. \n")
    }

    if (is.null(mtry)) {
        mtry = dim(X)[2]
        cat("mtry = p, use all variables. \n")
    }

    if (mtry > dim(X)[2]){
        mtry = dim(X)[2]
        cat("mtry cannot exceed p, set to mtry = p. \n")
    }

    if(p_categorical > dim(X)[2]){
        p_categorical = dim(X)[2]
        stop("p_categorical cannot exceed p")
    }
    # check input type

    check_positive_integer(num_sweeps, "num_sweeps")
    check_non_negative_integer(burnin, "burnin")
    check_non_negative_integer(p_categorical, "p_categorical")

    check_positive_integer(max_depth_m, "max_depth_m")
    check_positive_integer(Nmin_m, "Nmin_m")
    check_positive_integer(num_trees_m, "num_trees_m")
    check_positive_integer(num_cutpoints_m, "num_cutpoints_m")

    check_positive_integer(max_depth_v, "max_depth_v")
    check_positive_integer(Nmin_v, "Nmin_v")
    check_positive_integer(num_trees_v, "num_trees_v")
    check_positive_integer(num_cutpoints_v, "num_cutpoints_v")

    check_scalar(tau_m, "tau_m")
    check_scalar(no_split_penality_m, "no_split_penality_m")
    check_scalar(no_split_penality_v, "no_split_penality_v")
    check_scalar(alpha, "alpha")
    check_scalar(beta, "beta")
    check_scalar(kap, "kap")
    check_scalar(s, "s")


    obj = XBART_heterosk_cpp(y, X, Xtest, num_sweeps, burnin, p_categorical, mtry,
                             no_split_penality_m, num_trees_m, max_depth_m,
                             Nmin_m, num_cutpoints_m, tau_m,
                             no_split_penality_v, num_trees_v, max_depth_v,
                             Nmin_v, num_cutpoints_v, a_v, b_v, ini_var,
                             kap, s, tau_kap, tau_s, alpha, beta,
                             verbose, sampling_tau, parallel, set_random_seed,
                             random_seed, sample_weights_flag, nthread)

#    tree_json = r_to_json(mean(y), obj$model$tree_pnt)
#   obj$tree_json = tree_json

    class(obj) = "XBARTheteroskedastic"
    return(obj)
}