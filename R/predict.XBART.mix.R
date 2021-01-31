predict.XBART.mix <- function(model, X, Z) {
    # the model is y = theta * Z + g(X)

    # g(X)
    obj = .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, model$model_list$tree_pnt) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    obj = obj + Z %*% t(model$theta)

    return(obj)
}


