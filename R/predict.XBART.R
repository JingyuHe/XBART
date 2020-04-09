predict.XBART <- function(model, X) {
    obj = .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, model$model_list$tree_pnt)  # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}


predict.XBARTmultinomial <- function(model, X) {
    obj = .Call(`_XBART_xbart_multinomial_predict`, X, model$model_list$y_mean, model$num_class, model$model_list$tree_pnt)  # model$tree_pnt
    return(obj)
}

