predict.XBART <- function(object, X, ...) {
    out <- json_to_r(object$tree_json)

    obj <- .Call(`_XBART_xbart_predict`, X, object$model_list$y_mean, out$model_list$tree_pnt) # object$tree_pnt
    obj <- as.matrix(obj$yhats)
    return(obj)
}

predict_full <- function(object, X, ...) {
    out <- json_to_r(object$tree_json)
    obj <- .Call(`_XBART_xbart_predict_full`, X, object$model_list$y_mean, out$model_list$tree_pnt) # object$tree_pnt
    obj <- obj$yhats
    return(obj)
}

predict_gp <- function(object, y, X, Xtest, theta = 10, tau = 5, p_categorical = 0) {
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

    out <- json_to_r(object$tree_json)

    num_trees <- dim(object$sigma)[1]
    sigma <- as.matrix(object$sigma[num_trees, ])

    obj <- .Call(`_XBART_gp_predict`, y, X, Xtest, out$model_list$tree_pnt, object$residuals, sigma, theta, tau, p_categorical)

    obj <- obj$yhats_test
    return(obj)
}