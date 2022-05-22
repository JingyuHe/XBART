\name{predict.XBARTmultinomial}
\title{Predicting new observations using fitted XBART classification model.}
\alias{predict.XBARTmultinomial}
\description{
  This function predict testing data given fitted XBART classification model.
}
\usage{
\method{predict}{XBARTmultinomial}(object, X, iteration = NULL, burnin = 0, ...)
}
\arguments{
  \item{object}{Fitted \code{object} returned from XBART.multinomial fit.}
  \item{X}{A matrix of input testing data \eqn{X}.}
  \item{iteration}{...}
  \item{burnin}{The first \code{burnin} sweeps will be discarded for prediction.}
  \item{...}{optional parameters.}
}
\details{
XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. This function predicts label for the testing data.
}
\value{
  \item{prob}{A matrix of predicted probability of each class for the testing data. The number of rows is number of testing data, the number of columns is number of unique classes.}
  \item{label}{Predicted classes, the class with highest probability.}
}
\note{

}

\seealso{
  \code{\link{XBART.multinomial}}
}

\author{ Meijia Wang, Jingyu He }


\examples{


}


\keyword{XBART}
\keyword{classification}
\keyword{predict}
