predict_tree <- function(trees, Xnew, ...) {
    .Call(`_XBART_predict_tree`, trees, Xnew)
}

predict_tree_std <- function(trees, Xnew, ...) {
    .Call(`_XBART_predict_tree_std`, trees, Xnew)
}

# sample_int_ccrank <- function(n, size, prob) {
#     .Call(`_XBART_sample_int_ccrank`, n, size, prob)
# }

train_forest_root_std_all <- function(y, X, Xtest, M, L, N_sweeps, max_depth, Nmin, Ncutpoints, alpha, beta, tau, burnin = 1L, mtry = 0L, p_categorical = 0L, draw_sigma = FALSE, kap = 16, s = 4, verbose = FALSE, m_update_sigma = FALSE, draw_mu = FALSE, parallel = TRUE, random_seed = NULL, ...) {

    if(class(X) != "matrix"){
        cat("Input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }
    if(class(Xtest) != "matrix"){
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest = as.matrix(Xtest)
    }
    if(class(y) != "matrix"){
        cat("Input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    if(is.null(random_seed)){
        set_random_seed = FALSE
        random_seed = 0;
    }else{
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed = TRUE
    }

    .Call(`_XBART_train_forest_root_std_all`, y, X, Xtest, M, L, N_sweeps, max_depth, Nmin, Ncutpoints, alpha, beta, tau, burnin, mtry, p_categorical, draw_sigma, kap, s, verbose, m_update_sigma, draw_mu, parallel, set_random_seed, random_seed)
}



XBART <- function(y, X, Xtest, M, L, N_sweeps, max_depth, Nmin, Ncutpoints, alpha, beta, tau, burnin = 1L, mtry = 0L, p_categorical = 0L, draw_sigma = FALSE, kap = 16, s = 4, verbose = FALSE, m_update_sigma = FALSE, draw_mu = FALSE, parallel = TRUE, random_seed = NULL, ...) {

    if(class(X) != "matrix"){
        cat("Input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }
    if(class(Xtest) != "matrix"){
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest = as.matrix(Xtest)
    }
    if(class(y) != "matrix"){
        cat("Input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    if(is.null(random_seed)){
        set_random_seed = FALSE
        random_seed = 0;
    }else{
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed = TRUE
    }

    .Call(`_XBART`, y, X, Xtest, M, L, N_sweeps, max_depth, Nmin, Ncutpoints, alpha, beta, tau, burnin, mtry, p_categorical, draw_sigma, kap, s, verbose, m_update_sigma, draw_mu, parallel, set_random_seed, random_seed)
}