predict.XBARTmultinomial <- function(object, X, burnin = 0, ...) {
    
    
    out <- json_to_r(object$tree_json)
    obj <- .Call(`_XBART_xbart_multinomial_predict`, X, object$model_list$y_mean, object$num_class, out$model_list$tree_pnt) # object$tree_pnt
    
    num_sweeps <- dim(obj$yhats)[1]

    obj$prob <- apply(obj$yhats[burnin:num_sweeps, , ], c(2, 3), mean)
    obj$label <- apply(obj$prob, 1, which.max) - 1

    return(obj)
}