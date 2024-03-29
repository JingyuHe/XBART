% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.XBART.R
\name{predict_gp}
\alias{predict_gp}
\title{Predicting new observations using fitted XBART regression model, fit- ting Gaussian process to predict testing data out of the range of the training.}
\usage{
predict_gp(object, y, X, Xtest, theta = 10, tau = 5, p_categorical = 0)
}
\arguments{
\item{object}{Fitted \eqn{object} returned from XBART function.}

\item{y}{A vector of the training data \eqn{y}. Used to fit Gaussian process in the leaf nodes.}

\item{X}{A matrix of the training data \eqn{X}. Used to fit Gaussian process in the leaf nodes.}

\item{Xtest}{A matrix of the testing data to predict.}

\item{theta}{Lengthscale parameter of the covariance kernel.}

\item{tau}{Varaince parameter of the covariance kernel.}

\item{p_categorical}{Number of categorical \eqn{X} variables. All categorical variables should be placed after continuous variables.}
}
\value{
A vector of predictted outcome Y for the testing data.
}
\description{
This function predict testing data given fitted XBART regression model. It implements Gaussian process to predict testing data out of the range of the training (extrapolation).
}
\details{
This function fits Gaussian process in the leaf node, if the testing data lies out of the range of the training, the extrapolated prediction will be from Gaussian process. If the testing data lies within the range, the prediction is the same as that of predict.XBART.
}
