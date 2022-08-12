dump.XBCF <- function(model, file = "") {
    json_str = .Call(`_XBCF_r_to_json`, model$model_list$y_mean, model$model_list$tree_pnt)  # model$tree_pnt
    if (file == "") {
        return(json_str)
    } else {
        write(json_str, file)
    }
}