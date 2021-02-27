predict.XBART <- function(model, X) {
    
    out = json_to_r(model$tree_json)

    obj = .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, out$model_list$tree_pnt) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}


