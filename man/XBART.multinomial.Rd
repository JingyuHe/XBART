% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XBART.multinomial.R
\name{XBART.multinomial}
\alias{XBART.multinomial}
\title{XBART main function of XBART classification.}
\usage{
XBART.multinomial(
  y,
  num_class,
  X,
  num_trees = 20,
  num_sweeps = 20,
  max_depth = 20,
  Nmin = NULL,
  num_cutpoints = NULL,
  alpha = 0.95,
  beta = 1.25,
  tau_a = 1,
  tau_b = 1,
  no_split_penalty = NULL,
  burnin = 5,
  mtry = NULL,
  p_categorical = 0L,
  verbose = FALSE,
  parallel = TRUE,
  random_seed = NULL,
  sample_weights = TRUE,
  separate_tree = FALSE,
  weight = 1,
  update_weight = TRUE,
  update_tau = TRUE,
  nthread = 0,
  hmult = 1,
  heps = 0.1,
  ...
)
}
\arguments{
\item{y}{A vector of outcome variable of length n, expected to be discrete.}

\item{num_class}{Integer, number of different unique classes for the classification task.}

\item{X}{A matrix of input for the tree of size n by p. Column order matters: continuous features should all go before categorical. The number of categorical variables is p_categorical_con.}

\item{num_trees}{Integer, number of trees in the prognostic forest.}

\item{num_sweeps}{Integer, number of sweeps to fit for both forests.}

\item{max_depth}{Integer, maximum depth of the trees. The tree will stop grow if reaches the max depth.}

\item{Nmin}{Integer, minimal number of data points in a leaf. Any leaf will stop split if number of data points within is smaller than Nmin.}

\item{num_cutpoints}{Integer, number of cutpoint candidates to consider for each variable. Take in quantiles of the data.}

\item{alpha}{Scalar, BART prior parameter for trees. The default value is 0.95.}

\item{beta}{Scalar, BART prior parameter for trees. The default value is 1.25.}

\item{tau_a}{Scalar, prior of the leaf mean.}

\item{tau_b}{Scalar, prior of the leaf mean.}

\item{no_split_penalty}{Weight of no-split option. The default value is log(num_cutpoints), or you can take any other number in log scale.}

\item{burnin}{Integer, number of burnin sweeps.}

\item{mtry}{Integer, number of X variables to sample at each split of the tree.}

\item{p_categorical}{Integer, number of categorical variables in X, note that all categorical variables should be put after continuous variables. Default value is 0.}

\item{verbose}{Bool, whether to print fitting process on the screen or not.}

\item{random_seed}{Integer, random seed for replication.}

\item{sample_weights}{Bool, if TRUE, the weight to sample \eqn{X} variables at each tree will be sampled.}

\item{separate_tree}{Bool, if TRUE, fit separate trees for different classes, otherwise all classes share the same tree strucutre.}

\item{weight}{Replicate factor of the Poisson observations. The default value is 1.}

\item{update_weight}{Bool, if TRUE, sample the replicate factor to reflect the data entropy.}

\item{update_tau}{Bool, if TRUE, update the prior of leaf mean.}

\item{nthread}{Integer, number of threads to use if run in parallel.}

\item{hmult}{Prior of the replicate factor.}

\item{heps}{Prior of the replicate factor}

\item{...}{optional parameters to be passed to the low level function XBART}

\item{kap}{Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 16.}

\item{s}{Scalar, parameter of the inverse gamma prior on residual variance sigma^2. Default value is 4.}

\item{tau_kap}{Scalar, parameter of the inverse gamma prior on tau. Default value is 3.}

\item{tau_s}{Scalar, parameter of the inverse gamma prior on tau. Default value is 0.5.}

\item{paralll}{Bool, whether to run in parallel on multiple CPU threads.}
}
\value{
A list contains fitted trees as well as parameter draws at each sweep.
}
\description{
XBART main function of XBART classification.
}
\details{
XBART draws multiple samples of the forests (sweeps), each forest is an ensemble of trees. The final prediction is taking sum of trees in each forest, and average across different sweeps (with- out burnin sweeps). This function fits trees for multinomial classification tasks. Note that users have option to fit different tree structure for different classes, or let all classes share the same tree structure.
}
