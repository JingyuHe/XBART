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
#'
#' @details XBCF draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction returns predicted prognostic term, treatment effect and the final outcome \eqn{Y}
#' @return A list containing predicted prognostic term, treatment effect and final outcome \eqn{Y}.
#' @export


predict.XBCFdiscrete <- function(object, X_con, X_mod, Z, ...) {
    X_con <- as.matrix(X_con)
    X_mod <- as.matrix(X_mod)
    Z <- as.matrix(Z)
    out_con <- json_to_r(object$tree_json_con)
    out_mod <- json_to_r(object$tree_json_mod)
    obj <- .Call("_XBART_XBCF_discrete_predict", X_con, X_mod, Z, out_con$model_list$tree_pnt, out_mod$model_list$tree_pnt)
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
