#' Predicting new observations using fitted XBART (heteroskedastic) regression model.
#' @description This function predicts testing data given fitted XBART (heteroskedastic) regression model.
#' @param object Fitted \eqn{object} returned from XBART.heterosk function.
#' @param X A matrix of input testing data \eqn{X}
#'
#' @details XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (without burnin sweeps).
#' @return A vector of predicted mean component of outcome \eqn{Y} (mhats), a vector of predicted variance component of outcome \eqn{Y} (vhats).
#' @export


predict.XBARTheteroskedastic <- function(object, X, ...) {
    out_m <- json_to_r(object$tree_json_mean)
    out_v <- json_to_r(object$tree_json_variance)
    obj <- .Call(`_XBART_xbart_heteroskedastic_predict`, X,
                 out_m$model_list$tree_pnt, out_v$model_list$tree_pnt)
    return(obj)
}