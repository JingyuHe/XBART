#' Get post-burnin draws from trained model
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the prognostic term of size n by p1. Column order matters: continuos features should all bgo before of categorical.
#' @param x_mod An input matrix for the treatment term of size n by p2 (default x_mod = x_con). Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A list with two matrices. Each matrix corresponds to a set of draws of predicted values; rows are datapoints, columns are iterations.
#' @export
predict.XBCF <- function(model, x_con, x_mod=x_con, pihat=NULL, burnin=NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        x_con = as.matrix(x_con)
    }
    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }
    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }
    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj1 = .Call(`_XBART_xbart_predict`, x_con, model$model_list$tree_pnt_pr)
    obj2 = .Call(`_XBART_xbart_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- ncol(model$tauhats)
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    mus <- matrix(NA, nrow(x_con), sweeps - burnin)
    taus <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        taus[, i - burnin] = obj2$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
        mus[, i - burnin] = obj1$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    obj <- list(mudraws=mus, taudraws=taus)

    return(obj)
}

#' Get post-burnin draws from trained model (treatment term only)
#'
#' @param model A trained XBCF model.
#' @param x_mod An input matrix for the treatment term of size n by p2. Column order matters: continuos features should all go before categorical.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A matrix with a set of draws of predicted treatment effect estimates; rows are datapoints, columns are iterations.
#' @export
predictTauDraws <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    obj = .Call(`_XBART_xbart_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    return(tauhat.draws)
}

#' Get point-estimates of treatment effect
#'
#' @param model A trained XBCF model.
#' @param x_mod An input matrix for the treatment term of size n by p2. Column order matters: continuos features should all go before categorical.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return An array with point-estimates of treatment effect per datapoint in the given matrix.
#' @export
predictTaus <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    obj = .Call(`_XBART_xbart_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    tauhats <- rowMeans(tauhat.draws)

    return(tauhats)
}

#' Get post-burnin draws from trained model (prognostic term only)
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the treatment term of size n by p1. Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A matrix with a set of draws of predicted prognostic effect estimates; rows are datapoints, columns are iterations.
#' @export
predictMuDraws <- function(model, x_con, pihat=NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBART_xbart_predict`, x_con, model$model_list$tree_pnt_pr)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(x_con), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    return(muhat.draws)
}

#' Get point-estimates of prognostic effect
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the treatment term of size n by p1. Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return An array with point-estimates of prognostic effect per datapoint in the given matrix.
#' @export
predictMus <- function(model, x_con, pihat = NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBART_xbart_predict`, x_con, model$model_list$tree_pnt_pr)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(x_con), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    muhats <- rowMeans(muhat.draws)
    return(muhats)
}