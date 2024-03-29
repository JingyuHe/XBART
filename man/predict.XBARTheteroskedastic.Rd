% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.XBARTheteroskedastic.R
\name{predict.XBARTheteroskedastic}
\alias{predict.XBARTheteroskedastic}
\title{Predicting new observations using fitted XBART (heteroskedastic) regression model.}
\usage{
\method{predict}{XBARTheteroskedastic}(object, X, ...)
}
\arguments{
\item{object}{Fitted \eqn{object} returned from XBART.heterosk function.}

\item{X}{A matrix of input testing data \eqn{X}}
}
\value{
A vector of predicted mean component of outcome \eqn{Y} (mhats), a vector of predicted variance component of outcome \eqn{Y} (vhats).
}
\description{
This function predicts testing data given fitted XBART (heteroskedastic) regression model.
}
\details{
XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (without burnin sweeps).
}
