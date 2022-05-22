\name{XBART.multinomial}
\title{XBART multinomial classification}
\alias{XBART.multinomial}
\description{
    Main function to fit XBART model. Fit classification trees (categorical output).
}
\usage{
XBART.multinomial(y, num_class, X, num_trees = 20, num_sweeps = 20, max_depth = 20, 
Nmin = NULL, num_cutpoints = NULL, alpha = 0.95, beta = 1.25, tau_a = 1, 
tau_b = 1, no_split_penality = NULL, burnin = 5, mtry = NULL, 
p_categorical = 0L, verbose = FALSE, parallel = TRUE, random_seed = NULL, 
sample_weights = TRUE, separate_tree = FALSE, weight = 1, update_weight = TRUE, 
update_tau = TRUE, nthread = 0, hmult = 1, heps = 0.1, ...)
}
\arguments{
  \item{y}{\code{vector} of inputs \code{Y}. Response variable. }\
  \item{num_class}{Number of unique classes in \code{Y}.}
  \item{X}{\code{matrix} of inputs \code{X}. Regressors. }
  \item{num_trees}{Number of trees in each forest (sweep).}
  \item{num_sweeps}{Number of sweeps (samples of forests).}
  \item{max_depth}{Max depth of the tree. The tree will stop growing once reach maximum depth}
  \item{Nmin}{Minimal node size. The tree will stop growing once the number of data in the node is less than \eqn{Nmin}.}
  \item{num_cutpoints}{For continuous variable, number of adaptive cutpoint candidates considered in each split.}
  \item{alpha}{Parameter of the BART prior. The default value is 0.95.}
  \item{beta}{Parameter of the BART prior. The default value is 1.25.}
  \item{tau_a}{Prior of the leaf mean.}
  \item{tau_b}{Prior of the leaf mean.}
  \item{no_split_penality}{Weight of no-split option. The default value is log(num_cutpoints), or you can take any other numbers (should be in log scale).}
  \item{burnin}{Number of burn-in sweeps.}
  \item{mtry}{Number of varaibles considered in each split. Variables will be sampled similarly as Random Forest.}
  \item{p_categorical}{Number of categorical regressors in \eqn{X}. Note that all categorical variables should be put after all continuous variables.}
  \item{verbose}{Bool. If \code{TRUE}, print the progress on screen.}
  \item{parallel}{Bool. If \code{TRUE}, do computation in parallel.}
  \item{random_seed}{Random seed, should be a positive integer.}
  \item{sample_weights}{Bool. If \code{TRUE}, the weight to sample \eqn{X} variables at each tree will be sampled.}
  \item{separate_tree}{Bool. If \code{TRUE}, fit separate trees for different classes, otherwise all classes share the same tree structure.}
  \item{weight}{....}
  \item{update_weight}{...}
  \item{update_tau}{...}
  \item{nthread}{Number of threads to use if the \code{parallel=TRUE}.}
  \item{hmult}{...}
  \item{heps}{...}
  \item{...}{optional parameters to be passed to the low level function \code{XBART}.}
}
\details{
XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (without burnin sweeps). This function fits trees for multinomial classification tasks. Note that users have option to fit different tree structure for different classes, or let all classes share the same tree structure.
}
\value{
  \item{num_class}{Number of unique classes in \eqn{Y}.}
  \item{yhats_train}{Fitted \eqn{Y} of the training data.}
  \item{weights}{Weights of samples.}
  \item{tau_a}{Samples of tau_a.}
  \item{lambda}{Samples of \eqn{lambda}.}
  \item{importance}{Variable importance of all \eqn{X} variables, based on counts being used as split rules.}
  \item{depth}{Depth of each tree.}
  \item{model_list}{A list objects of fitted tree, including a pointer to the tree structure in the memory.}
  \item{treedraws}{Fitted tree saved in text.}
  \item{tree_json}{Fitted tree saved in json text format.}
  \item{separate_tree}{Bool. Indicating whether fitting separate tree structures for different classes or not.}
}
\note{

}

\seealso{
  \code{\link{XBART}}
}

\references{
He, Jingyu, Saar Yalov, and P. Richard Hahn. "XBART: Accelerated Bayesian additive regression trees." The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.

He, Jingyu, and P. Richard Hahn. "Stochastic tree ensembles for regularized nonlinear regression." Journal of the American Statistical Association (2021): 1-20.
}

\author{ Meijia Wang, Jingyu He }


\examples{


}


\keyword{XBART}
\keyword{classification}


