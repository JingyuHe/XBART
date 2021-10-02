predict.XBART <- function(model, X, distance_s=1) {
    
    out = json_to_r(model$tree_json)

    obj = .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, out$model_list$tree_pnt, distance_s) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}


