library(XBART)

cen <- c(20, 50, 80)
cf <- c(3.55, 3, 2.45)

opt <- 1

### GENERATE DATA ###
n <- 2000
x1 <- rnorm(n)
x2 <- runif(n, 1, 2)
x3 <- runif(n, 0, 1)
x4 <- rbinom(n, 1, 0.6)

x <- data.frame(x1, x2, x3, x4)

# set time limit (based on Yobs distribution), step for tgrid
limit <- 20
step <- 0.1
tgrid <- seq(0, limit, step)


mu <- 2.5 + 0.2 * x[, 2] + 0.1 * x[, 4]
sig <- 0.3 + 0.05 * abs(x[, 1])
Y <- rlnorm(n, mu, sig)
# hist(Y,30)

C <- rlnorm(n, cf[opt] - 0.1 * x[, 2], 0.5 + 0.05 * abs(x[, 1]))
# hist(C,30)

Yobs <- pmin(Y, C)
delta <- as.numeric(Y < C)

rmst.true <- rep(0, nrow(x))
ntest <- 100
for (i in 1:ntest) {
  rmst.true[i] <- integrate(plnorm, 0, limit, meanlog = mu[i], sdlog = sig[i], lower.tail = FALSE)$value
}

# convert to log scale
logt <- log(Yobs)


num_sweeps = 30
burnin = 10


fit = XBART.heterosk(
  y = matrix(logt), X = x, Xtest = x,
  num_sweeps = num_sweeps,
  burnin = burnin,
  p_categorical = 0,
  mtry = 1,
  num_trees_m = 20,
  max_depth_m = 250,
  Nmin_m = 1,
  num_cutpoints_m = 20,
  num_trees_v = 5,
  max_depth_v = 10,
  Nmin_v = 50,
  num_cutpoints_v = 100,
  ini_var = 1,
  verbose = FALSE,
  parallel = FALSE,
  sample_weights_flag = FALSE
)


pred1 = rowMeans(predict(fit, as.matrix(x))$mhats[,burnin:num_sweeps])




fit2 = XBART.survival(
  y = matrix(logt), X = x, tau = delta,
  num_sweeps = num_sweeps,
  burnin = burnin,
  p_categorical = 0,
  mtry = 1,
  num_trees_m = 20,
  max_depth_m = 250,
  Nmin_m = 1,
  num_cutpoints_m = 20,
  num_trees_v = 5,
  max_depth_v = 10,
  Nmin_v = 50,
  num_cutpoints_v = 100,
  ini_var = 1,
  verbose = FALSE,
  parallel = FALSE,
  sample_weights_flag = FALSE
)

pred2 = rowMeans(predict(fit2, as.matrix(x))$mhats[,burnin:num_sweeps])
