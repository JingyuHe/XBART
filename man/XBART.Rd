\name{XBART}
\alias{XBART}
\title{XBART: Accelerated Bayesian Additive Regression Trees}
\description{
    Main function to fit XBART model. Fit a regression tree (continuous output).
}
\usage{
\method{XBART}{default}(y, X, Xtest, num_trees, num_sweeps, max_depth, Nmin, num_cutpoints, 
alpha, beta, tau, no_split_penality = "Auto", burnin = 1L, mtry = 0L, p_categorical = 0L, 
kap = 16, s = 4, verbose = FALSE, parallel = TRUE, random_seed = NULL, ...)
}
\arguments{
  \item{y}{\code{matrix} of inputs \code{Y}. Response variable. }
  \item{X}{\code{matrix} of inputs \code{X}. Regressors. }
  \item{Xtest}{\code{matrix} of test set.}
  \item{num_trees}{Number of trees.}
  \item{num_sweeps}{Number of sweeps.}
  \item{max_depth}{\code{matrix}, \code{N_sweeps} by \code{M}. Indicate max depth of each tree in each sweep.}
  \item{Nmin}{Minimal node size.}
  \item{num_cutpoints}{For continuous variable, number of adaptive cutpoint candidates considered in each split.}
  \item{alpha}{Prior parameter of BART.}
  \item{beta}{Prior parameter of BART.}
  \item{tau}{Prior parameter of BART.}
  \item{no_split_penality}{Weight of no-split option. The default value is log(num_cutpoints), or you can take any other numbers (should be in log scale).}
  \item{burnin}{Number of burn-in sweeps.}
  \item{mtry}{Number of varaibles considered in each split. Like random forest.}
  \item{p_categorical}{Number of categorical regressors. (\code{X} and \code{Xtest}).}
  \item{kap}{Prior parameter of \eqn{\sigma}.}
  \item{s}{Prior parameter of \eqn{\sigma}.}
  \item{verbose}{If \code{TRUE}, print the progress on screen.}
  \item{parallel}{If \code{TRUE}, do computation in parallel.}
  \item{random_seed}{Random seed, should be a positive integer.}
  \item{...}{optional parameters to be passed to the low level function \code{XBART}.}
}
\details{
  For details of the model, please see the original paper XBART: Accelerated Bayesian Additive Regression Trees.
}
\value{
  \item{loops}{A \code{vector} of number of elliptical slice sampler loops for each posterior sample.}
  \item{sigma}{A \code{vector} of posterior samples of residual standard error.}
  \item{vglobal}{A \code{vector} of posterior samples of the global shrinkage parameter.}
  \item{beta}{A \code{matrix} of posterior samples of coefficients.}
  \item{fitted.values}{Fitted values of the regression model. Take posterior mean of coefficients with 20\% burnin samples.}
  \item{residuals}{Residuals of the regression model, equals \code{y - fitted.values}.}
}
\note{
  \code{horseshoe} is essentially call function \code{bayeslm} with \code{prior = "horseshoe"}. Same for \code{sharkfin}, \code{ridge}, \code{blasso}, \code{nonlocal}.
}

\references{
  Hahn, P. Richard, Jingyu He, and Hedibert Lopes. \emph{Efficient sampling for Gaussian linear regression with arbitrary priors.} (2017).
}

\author{ Jingyu He \email{jingyu.he@chicagobooth.edu} }


\examples{


}


\keyword{XBART}