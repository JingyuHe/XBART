\name{predict.XBART}
\title{Predicting new observations using fitted XBART regression model.}
\alias{predict.XBART}
\description{
  This function predict testing data given fitted XBART regression model.
}
\usage{
\method{predict}{XBART}(object, X, ...)
}
\arguments{
  \item{object}{Fitted \code{object} returned from XBART fit.}
  \item{X}{A matrix of input testing data \eqn{X}.}
  \item{...}{optional parameters.}
}
\details{
XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (without burnin sweeps).
}
\value{
  \item{yhats}{A vector of predictted outcome \eqn{Y} for the testing data.}
}
\note{

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
