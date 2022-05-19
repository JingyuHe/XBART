predict.XBART <- function(model, X) {
    out <- json_to_r(model$tree_json)

    obj <- .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, out$model_list$tree_pnt) # model$tree_pnt
    obj <- as.matrix(obj$yhats)
    return(obj)
}

predict.full <- function(model, X) {
    out <- json_to_r(model$tree_json)

    obj <- .Call(`_XBART_xbart_predict_full`, X, model$model_list$y_mean, out$model_list$tree_pnt) # model$tree_pnt
    obj <- obj$yhats
    return(obj)
}

predict.gp <- function(model, y, X, Xtest, theta = 10, tau = 5, p_categorical = 0) {
    if (!("matrix" %in% class(X))) {
        cat("Input X is not a matrix, try to convert type.\n")
        X <- as.matrix(X)
    }

    if (!("matrix" %in% class(Xtest))) {
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest <- as.matrix(Xtest)
    }

    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    out <- json_to_r(model$tree_json)

    num_trees <- dim(model$sigma)[1]
    sigma <- as.matrix(model$sigma[num_trees, ])

    obj <- .Call(`_XBART_gp_predict`, y, X, Xtest, out$model_list$tree_pnt, model$residuals, sigma, theta, tau, p_categorical)

    obj <- obj$yhats_test
    return(obj)
}