#' Predicting new observations using fitted XBART regression model.
#' @description This function predicts testing data given fitted XBART regression model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X A matrix of input testing data \eqn{X}
#'
#' @details XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (without burnin sweeps).
#' @return yhats A vector of predictted outcome \eqn{Y} for the testing data.
#' @export


predict.XBART <- function(object, X, ...) {
    out <- json_to_r(object$tree_json)
    obj <- .Call(`_XBART_xbart_predict`, X, object$model_list$y_mean, out$model_list$tree_pnt) # object$tree_pnt
    obj <- as.matrix(obj$yhats)
    return(obj)
}

#' Predicting new observations using fitted XBCF continuous treatment model.
#' @description This function predicts testing data given fitted XBCF continuous treatment model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X_con A matrix of input testing data for the prognostic forest.
#' @param X_mod A matrix of input testing data for the treatment forest.
#' @param Z A vector of input testing data for the treatment variable.
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect and final outcome \eqn{Y}.
#' @export


predict.XBCFcontinuous <- function(object, X_con, X_mod, Z, ...) {
    X_con <- as.matrix(X_con)
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    obj <- .Call("_XBART_XBCF_continuous_predict", X_con, X_mod, Z, out_con$model_list$tree_pnt, out_mod$model_list$tree_pnt)
    return(obj)
}

#' Predicting new observations using fitted XBCF binary treatment model.
#' @description This function predicts testing data given fitted XBCF continuous treatment model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X_con A matrix of input testing data for the prognostic forest.
#' @param X_mod A matrix of input testing data for the treatment forest.
#' @param Z A vector of input testing data for the treatment variable.
#' @param pihat An array of propensity score estimates.
#' @param burnin The number of burn-in iterations to discard from averaging (the default value is 0).
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect and final outcome \eqn{Y}.
#' @export


predict.XBCFdiscrete <- function(object, X_con, X_mod, Z, pihat = NULL, burnin = 0L, ...) {
    stopifnot("Propensity scores (pihat) must be provided by user for prediction." = !is.null(pihat))

    X_con <- as.matrix(cbind(pihat, X_con))
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    obj <- .Call("_XBART_XBCF_discrete_predict", X_con, X_mod, Z, out_con$model_list$tree_pnt, out_mod$model_list$tree_pnt)

    burnin <- burnin
    sweeps <- nrow(object$a)
    mus <- matrix(NA, nrow(X_con), sweeps)
    taus <- matrix(NA, nrow(X_mod), sweeps)
    seq <- c(1:sweeps)
    for (i in seq) {
        taus[, i] <- obj$tau[, i] * object$sdy * (object$b[i, 2] - object$b[i, 1])
        mus[, i] <- object$sdy * (obj$mu[, i] * (object$a[i]) + obj$tau[, i] * object$b[i, 1]) + object$meany
    }

    obj$tau.adj <- taus
    obj$mu.adj <- mus
    obj$yhats.adj <- Z[, 1] * obj$tau.adj + obj$mu.adj
    obj$tau.adj.mean <- rowMeans(obj$tau.adj[, (burnin + 1):sweeps])
    obj$mu.adj.mean <- rowMeans(obj$mu.adj[, (burnin + 1):sweeps])
    obj$yhats.adj.mean <- rowMeans(obj$yhats.adj[, (burnin + 1):sweeps])

    return(obj)
}

predict_full <- function(object, X, ...) {
    out <- json_to_r(object$tree_json)
    obj <- .Call(`_XBART_xbart_predict_full`, X, object$model_list$y_mean, out$model_list$tree_pnt) # object$tree_pnt
    obj <- obj$yhats
    return(obj)
}

#' Predicting new observations using fitted XBART regression model, fit- ting Gaussian process to predict testing data out of the range of the training.
#' @description This function predict testing data given fitted XBART regression model. It implements Gaussian process to predict testing data out of the range of the training (extrapolation).
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param y A vector of the training data \eqn{y}. Used to fit Gaussian process in the leaf nodes.
#' @param X A matrix of the training data \eqn{X}. Used to fit Gaussian process in the leaf nodes.
#' @param Xtest A matrix of the testing data to predict.
#' @param theta Lengthscale parameter of the covariance kernel.
#' @param tau Varaince parameter of the covariance kernel.
#' @param p_categorical Number of categorical \eqn{X} variables. All categorical variables should be placed after continuous variables.
#'
#' @details This function fits Gaussian process in the leaf node, if the testing data lies out of the range of the training, the extrapolated prediction will be from Gaussian process. If the testing data lies within the range, the prediction is the same as that of predict.XBART.
#' @return A vector of predictted outcome Y for the testing data.
#'
predict_gp <- function(object, y, X, Xtest, theta = 10, tau = 5, p_categorical = 0) {
    if (!("matrix" %in% class(X))) {
        cat("Input X is not a matrix, try to convert type.\n")
        X <- as.matrix(X)
    }

    if (!("matrix" %in% class(Xtest))) {
        cat("Input Xtest is not a matrix, try to convert type.\n")
        Xtest <- as.matrix(Xtest)
    }

    if (!("matrix" %in% class(y))) {
        cat("Input y is not a matrix, try to convert type.\n")
        y <- as.matrix(y)
    }

    out <- json_to_r(object$tree_json)

    num_trees <- dim(object$sigma)[1]
    sigma <- as.matrix(object$sigma[num_trees, ])

    obj <- .Call(`_XBART_gp_predict`, y, X, Xtest, out$model_list$tree_pnt, object$residuals, sigma, theta, tau, p_categorical)

    obj <- obj$yhats_test
    return(obj)
}

#' Predicting new observations using fitted XBCF binary treatment model with heteroskedastic variance.
#' @description This function predicts testing data given fitted XBCF binary treatment model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X_con A matrix of input testing data for the prognostic forest.
#' @param X_mod A matrix of input testing data for the treatment forest.
#' @param Z A vector of input testing data for the treatment variable.
#' @param pihat An array of propensity score estimates.
#' @param burnin The number of burn-in iterations to discard from averaging (the default value is 0).
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect, standard deviation and final outcome \eqn{Y}.
#' @export


predict.XBCFdiscreteHeterosk <- function(object, X_con, X_mod, Z, pihat = NULL, burnin = 0L, ...) {
    stopifnot("Propensity scores (pihat) must be provided by user for prediction." = !is.null(pihat))

    X_con <- as.matrix(cbind(pihat, X_con))
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    out_v <- json_to_r(object$tree_json_v)
    obj <- .Call(
        "_XBART_XBCF_discrete_heteroskedastic_predict", X_con, X_mod, Z,
        out_con$model_list$tree_pnt,
        out_mod$model_list$tree_pnt,
        out_v$model_list$tree_pnt
    )

    burnin <- burnin
    sweeps <- nrow(object$a)
    mus <- matrix(NA, nrow(X_con), sweeps)
    taus <- matrix(NA, nrow(X_mod), sweeps)
    seq <- c(1:sweeps)
    for (i in seq) {
        taus[, i] <- obj$tau[, i] * (object$b[i, 2] - object$b[i, 1])
        mus[, i] <- obj$mu[, i] * (object$a[i]) + object$meany + obj$tau[, i] * object$b[i, 1]
    }

    obj$variance <- obj$variance * object$sdy
    obj$tau.adj <- taus
    obj$mu.adj <- mus
    obj$yhats.adj <- Z[, 1] * obj$tau.adj + obj$mu.adj
    obj$tau.adj.mean <- rowMeans(obj$tau.adj[, (burnin + 1):sweeps])
    obj$mu.adj.mean <- rowMeans(obj$mu.adj[, (burnin + 1):sweeps])
    obj$yhats.adj.mean <- rowMeans(obj$yhats.adj[, (burnin + 1):sweeps])

    return(obj)
}



#' Predicting new observations using fitted XBCF binary treatment model with heteroskedastic variance.
#' @description This function predicts testing data given fitted XBCF binary treatment model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X_con A matrix of input testing data for the prognostic forest.
#' @param X_mod A matrix of input testing data for the treatment forest.
#' @param Z A vector of input testing data for the treatment variable.
#' @param pihat An array of propensity score estimates.
#' @param burnin The number of burn-in iterations to discard from averaging (the default value is 0).
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect, standard deviation and final outcome \eqn{Y}.
#' @export


predict.XBCFdiscreteHeterosk2 <- function(object, X_con, X_mod, Z, pihat = NULL, burnin = 0L, ...) {
    stopifnot("Propensity scores (pihat) must be provided by user for prediction." = !is.null(pihat))

    X_con <- as.matrix(cbind(pihat, X_con))
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    out_v_con <- json_to_r(object$tree_json_v_con)
    out_v_trt <- json_to_r(object$tree_json_v_trt)

    obj <- .Call(
        "_XBART_XBCF_discrete_heteroskedastic_predict2", X_con, X_mod, Z,
        out_con$model_list$tree_pnt,
        out_mod$model_list$tree_pnt,
        out_v_con$model_list$tree_pnt,
        out_v_trt$model_list$tree_pnt
    )
    burnin <- burnin
    sweeps <- nrow(object$a)
    mus <- matrix(NA, nrow(X_con), sweeps)
    taus <- matrix(NA, nrow(X_mod), sweeps)
    seq <- c(1:sweeps)
    for (i in seq) {
        taus[, i] <- obj$tau[, i] * (object$b[i, 2] - object$b[i, 1])
        mus[, i] <- obj$mu[, i] * (object$a[i]) + object$meany + obj$tau[, i] * object$b[i, 1]
    }

    obj$variance <- obj$variance * object$sdy
    obj$tau.adj <- taus
    obj$mu.adj <- mus
    obj$yhats.adj <- Z[, 1] * obj$tau.adj + obj$mu.adj
    obj$tau.adj.mean <- rowMeans(obj$tau.adj[, (burnin + 1):sweeps])
    obj$mu.adj.mean <- rowMeans(obj$mu.adj[, (burnin + 1):sweeps])
    obj$yhats.adj.mean <- rowMeans(obj$yhats.adj[, (burnin + 1):sweeps])

    return(obj)
}



#' Predicting new observations using fitted XBCF binary treatment model with heteroskedastic variance.
#' @description This function predicts testing data given fitted XBCF binary treatment model.
#' @param object Fitted \eqn{object} returned from XBART function.
#' @param X_con A matrix of input testing data for the prognostic forest.
#' @param X_mod A matrix of input testing data for the treatment forest.
#' @param Z A vector of input testing data for the treatment variable.
#' @param pihat An array of propensity score estimates.
#' @param burnin The number of burn-in iterations to discard from averaging (the default value is 0).
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect, standard deviation and final outcome \eqn{Y}.
#' @export


predict.XBCFdiscreteHeterosk3 <- function(object, X_con, X_mod, Z, pihat = NULL, burnin = 0L, ...) {
    stopifnot("Propensity scores (pihat) must be provided by user for prediction." = !is.null(pihat))

    X_con <- as.matrix(cbind(pihat, X_con))
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    out_v_con <- json_to_r(object$tree_json_v_con)
    out_v_mod <- json_to_r(object$tree_json_v_mod)

    obj <- .Call(
        "_XBART_XBCF_discrete_heteroskedastic_predict3", X_con, X_mod, Z,
        out_con$model_list$tree_pnt,
        out_mod$model_list$tree_pnt,
        out_v_con$model_list$tree_pnt,
        out_v_mod$model_list$tree_pnt
    )
    burnin <- burnin
    sweeps <- nrow(object$a)
    mus <- matrix(NA, nrow(X_con), sweeps)
    taus <- matrix(NA, nrow(X_mod), sweeps)
    seq <- c(1:sweeps)
    for (i in seq) {
        taus[, i] <- obj$tau[, i] * (object$b[i, 2] - object$b[i, 1])
        mus[, i] <- obj$mu[, i] * (object$a[i]) + object$meany + obj$tau[, i] * object$b[i, 1]
    }

    obj$variance <- obj$variance * object$sdy
    obj$variance_con <- obj$variance * object$sdy
    obj$variance_mod <- obj$variance
    obj$tau.adj <- taus
    obj$mu.adj <- mus
    obj$yhats.adj <- Z[, 1] * obj$tau.adj + obj$mu.adj
    obj$tau.adj.mean <- rowMeans(obj$tau.adj[, (burnin + 1):sweeps])
    obj$mu.adj.mean <- rowMeans(obj$mu.adj[, (burnin + 1):sweeps])
    obj$yhats.adj.mean <- rowMeans(obj$yhats.adj[, (burnin + 1):sweeps])

    return(obj)
}
