\name{predict_gp}
\title{Predicting new observations using fitted XBART regression model, fitting Gaussian process to predict testing data out of the range of the training.}
\alias{predict_gp}
\description{
  This function predict testing data given fitted XBART regression model. It implements Gaussian process to predict testing data out of the range of the training (extrapolation).
}
\usage{
predict_gp(object, y, X, Xtest, theta = 10, tau = 5, p_categorical = 0)
}
\arguments{
  \item{object}{Fitted \code{object} returned from XBART fit.}
  \item{y}{A vector of the training data \eqn{y}. Used to fit Gaussian process in the leaf nodes.}
  \item{X}{A matrix of the training data \eqn{X}. Used to fit Gaussian process in the leaf nodes.}
  \item{Xtest}{A matrix of the testing data to predict.}
  \item{theta}{....}
  \item{tau}{....}
  \item{p_categorical}{Number of categorical \eqn{X} variables. All of categorical variables should be placed after continuous ones.}
}
\details{
This function fits Gaussian process in the leaf node, if the testing data lies out of the range of the training, the extrapolated prediction will be from Gaussian process. If the testing data lies within the range, the prediction is the same as that of predict.XBART.
}
\value{
  \item{yhats}{A vector of predictted outcome \eqn{Y} for the testing data.}
}
\note{

}
\references{
  Wang, Meijiang, Jingyu He, and P. Richard Hahn. "Local Gaussian process extrapolation for BART models with applications to causal inference." arXiv preprint arXiv:2204.10963 (2022).
}
\seealso{
  \code{\link{XBART}}
}

\author{ Jingyu He }


\examples{


}


\keyword{XBART}
\keyword{regression}
\keyword{predict}
