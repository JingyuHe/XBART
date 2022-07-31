XBCF.continuous <- function(y, Z, X_con, X_mod, num_trees_con, num_trees_mod, num_sweeps, max_depth = 250,
                            Nmin = 1, num_cutpoints = 100, alpha_con = 0.95, beta_con = 1.25, alpha_mod = 0.95, beta_mod = 1.25, tau_con = NULL, tau_mod = NULL,
                            no_split_penality = NULL, burnin = 1L, mtry_con = NULL, mtry_mod = NULL, p_categorical_con = 0L, p_categorical_mod = 0L,
                            kap = 16, s = 4, tau_con_kap = 3, tau_con_s = 0.5, tau_mod_kap = 3, tau_mod_s = 0.5, verbose = FALSE, sampling_tau = TRUE, parallel = TRUE, random_seed = NULL,
                            sample_weights_flag = TRUE, nthread = 0, ...) {
    if (!("matrix" %in% class(X_con))) {
        cat("Input X_con is not a matrix, try to convert type.\n")
        X_con <- as.matrix(X_con)
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

    if (is.null(no_split_penality) || no_split_penality == "Auto") {
        no_split_penality <- log(num_cutpoints)
    }

    if (is.null(tau_con)) {
        tau_con <- 1 / num_trees_con
        cat("tau_con = 1/num_trees_con, default value. \n")
    }

    if (is.null(tau_mod)) {
        tau_mod <- 1 / num_trees_mod
        cat("tau_mod = 1/num_trees_mod, default value. \n")
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
    check_scalar(no_split_penality, "no_split_penality")
    check_scalar(alpha_con, "alpha_con")
    check_scalar(beta_con, "beta_con")
    check_scalar(alpha_mod, "alpha_mod")
    check_scalar(beta_mod, "beta_mod")
    check_scalar(kap, "kap")
    check_scalar(s, "s")

    obj <- XBCF_continuous_cpp(y, Z, X_con, X_mod, num_trees_con, num_trees_mod, num_sweeps, max_depth, Nmin, num_cutpoints, alpha_con, beta_con, alpha_mod, beta_mod, tau_con, tau_mod, no_split_penality, burnin, mtry_con, mtry_mod, p_categorical_con, p_categorical_mod, kap, s, tau_con_kap, tau_con_s, tau_mod_kap, tau_mod_s, verbose, sampling_tau, parallel, set_random_seed, random_seed, sample_weights_flag, nthread)

    class(obj) <- "XBCF"
    return(obj)
}
