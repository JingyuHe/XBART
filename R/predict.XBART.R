predict.XBART <- function(model, X, distance_s=1) {
    
    out = json_to_r(model$tree_json)

    obj = .Call(`_XBART_xbart_predict`, X, model$model_list$y_mean, out$model_list$tree_pnt, distance_s) # model$tree_pnt
    obj = as.matrix(obj$yhats)
    return(obj)
}


# predict.gp <- function() {
#     # structure for returning training data in each leaf for each test dp.
#     n_test = 10
#     n_trees = 10
#     n_sweeps = 100
#     ret = list()
#     for (i in 1:n_test){
#         ret[[i]] = list()
#         for(j in 1:(n_trees * n_sweeps)){
#             ret[[i]][[j]] = .Call(`_XBART_gp_predict`)
#         }
#     }
   
#     return(ret)
# }
