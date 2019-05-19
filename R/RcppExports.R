# predict_tree <- function(trees, Xnew, ...) {
#     .Call(`_XBART_predict_tree`, trees, Xnew)
# }

# predict_tree_std <- function(trees, Xnew, ...) {
#     .Call(`_XBART_predict_tree_std`, trees, Xnew)
# }

# sample_int_ccrank <- function(n, size, prob) {
#     .Call(`_XBART_sample_int_ccrank`, n, size, prob)
# }

# train_forest_root_std_all <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality = "Auto", burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, random_seed = NULL, ...) {

#     if(class(X) != "matrix"){
#         cat("Input X is not a matrix, try to convert type.\n")
#         X = as.matrix(X)
#     }
#     if(class(Xtest) != "matrix"){
#         cat("Input Xtest is not a matrix, try to convert type.\n")
#         Xtest = as.matrix(Xtest)
#     }
#     if(class(y) != "matrix"){
#         cat("Input y is not a matrix, try to convert type.\n")
#         y = as.matrix(y)
#     }

#     if(burnin >= num_sweeps){
#         cat("Burnin samples should be smaller than number of sweeps.\n")
#         return();
#     }

#     if(is.null(random_seed)){
#         set_random_seed = FALSE
#         random_seed = 0;
#     }else{
#         cat("Set random seed as ", random_seed, "\n")
#         set_random_seed = TRUE
#     }
#     if(no_split_penality == "Auto"){
#         no_split_penality = log(num_cutpoints)
#     }

#     .Call(`_XBART_train_forest_root_std_all`, y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality,burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed)
# }



XBART <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality = "Auto", burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, random_seed = NULL,sample_weights_flag = TRUE ,...) {

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

    if(burnin >= num_sweeps){
        cat("Burnin samples should be smaller than number of sweeps.\n")
        return();
    }
    if(no_split_penality == "Auto"){
        no_split_penality = log(num_cutpoints)
    }

    obj = .Call(`_XBART`, y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau,no_split_penality, burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed,sample_weights_flag)
    class(obj) = "XBART"
    return(obj)
}

XBART.CLT <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality = "Auto", burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, random_seed = NULL, sample_weights_flag = TRUE, ...) {

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

    if(burnin >= num_sweeps){
        cat("Burnin samples should be smaller than number of sweeps.\n")
        return();
    }
    if(no_split_penality == "Auto"){
        no_split_penality = log(num_cutpoints)
    }
    obj = .Call(`_XBART_CLT`, y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality,burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed, sample_weights_flag)
    class(obj) = "XBART" # Change to XBARTProbit?
    return(obj)
}

XBART.Probit <- function(y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality = "Auto",burnin = 1L, mtry = 0L, p_categorical = 0L, kap = 16, s = 4, verbose = FALSE, parallel = TRUE, random_seed = NULL, sample_weights_flag = TRUE,...) {

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

    if(burnin >= num_sweeps){
        cat("Burnin samples should be smaller than number of sweeps.\n")
        return();
    }
    if(no_split_penality == "Auto"){
        no_split_penality = 0
    }
    obj = .Call(`_XBART_Probit`, y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, alpha, beta, tau, no_split_penality,burnin, mtry, p_categorical, kap, s, verbose, parallel, set_random_seed, random_seed,sample_weights_flag)
    class(obj) = "XBART" # Change to XBARTProbit?
    return(obj)
}



predict.XBART <- function(model, X) {
    obj = .Call(`_xbart_predict`, X, model$model_list$y_mean, model$model_list$tree_pnt) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}

load.XBART <- function(fileName){
    json_str = readChar(fileName, file.info(fileName)$size)
    obj = .Call(`_json_to_r`,json_str) # model$tree_pnt
    class(obj) = "XBART"
    return(obj)
}

dump.XBART <- function(model,file = ""){
    json_str = .Call(`_r_to_json`, model$model_list$y_mean, model$model_list$tree_pnt) # model$tree_pnt
    if(file == ""){
        return(json_str)
    }else{
        write(json_str,file)
    }
}
