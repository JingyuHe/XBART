predict.XBARTmultinomial <- function(model, X, iteration = NULL) {

    if (is.null(iteration)) {
        cat("Predict with all iterations.", "\n")
        iteration = 0:(model$model_list$num_sweeps - 1)
    } else {
        ## C++ counts from 0, subtract 1 to match the index
        ## 1L means integer 1, rather than a float 1
        iteration = iteration - 1L
    }

    if (!is.integer(iteration)) {
        stop("Iteration index has to be a vector of integers.")
    }

    # check whether iteration is out of bound
    test = (iteration >= 0L) * (iteration <= model$model_list$num_sweeps)
    if (sum(test) != length(iteration)) {
        stop("Index of iteration is out of bound.")
    }

    if(model$separate_tree){
        out = json_to_r_3D(model$tree_json)
        obj = .Call(`_XBART_xbart_multinomial_predict_3D`, X, model$model_list$y_mean, model$num_class, out$model_list$tree_pnt, iteration) # model$tree_pnt
    }else{
        out = json_to_r(model$tree_json)
        obj = .Call(`_XBART_xbart_multinomial_predict`, X, model$model_list$y_mean, model$num_class, out$model_list$tree_pnt, iteration) # model$tree_pnt
    }   

    return(obj)
}