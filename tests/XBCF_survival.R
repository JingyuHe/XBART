library(XBART)
library(nftbart)

set.seed(218390)

samp_nft <- TRUE
ini_impute <- 0.5

cen <- c(20, 50, 80)
cf <- c(3.55, 3, 2.45)

opt <- 2

par(mfrow = c(1, 3))

### GENERATE DATA ###
n <- 3000
x1 <- rnorm(n)
x2 <- runif(n, 1, 2)
x3 <- runif(n, 0, 1)
x4 <- rbinom(n, 1, 0.6)
x5 <- rbinom(n, 1, 0.7)

x <- data.frame(x1, x2, x3, x4, x5)

mu <- 2.5 + 0.2 * x[, 2] + 0.1 * x[, 4]

# define treatment effects
tau <- 3 + 0.5 * x[, 4] * (2 * x[, 5] - 1)

# compute propensity scores and treatment assignment
pi <- pnorm(-0.5 + mu - x[, 2] + 0. * x[, 4], 0, 3)

z <- rbinom(n, 1, pi)

Ey <- mu + tau * z

sig <- 0.3 + 0.05 * abs(x[, 1])

Y <- rlnorm(n, Ey, sig)

C <- rlnorm(n, cf[opt] - 0.1 * x[, 2], 0.5 + 0.05 * abs(x[, 1]))
# hist(C,30)

Yobs <- pmin(Y, C)
delta <- as.numeric(Y < C)


# set time limit (based on Yobs distribution), step for tgrid
limit <- quantile(Yobs, 0.9)
step <- 0.5
tgrid <- seq(0, limit, step)


ntest <- 100
rmst.true <- rep(0, ntest)
for (i in 1:ntest) {
    rmst.true[i] <- integrate(plnorm, 0, limit, meanlog = mu[i], sdlog = sig[i], lower.tail = FALSE)$value
}



# convert to log scale
logt <- log(Yobs)


num_sweeps <- 150
burnin <- 15

pihat <- pi
a_scaling = TRUE
b_scaling <- TRUE

fit <- XBCF.survival.discrete.heterosk3(
    y = matrix(logt) - mean(logt),
    Z = z, delta = delta, X_con = x, X_mod = x, pihat = pihat,
    p_categorical_con = 5, p_categorical_mod = 5,
    num_trees_con = 15, num_trees_mod = 5,
    num_sweeps = num_sweeps, burnin = burnin, sample_weights = TRUE,
    a_scaling = a_scaling, b_scaling = b_scaling
)
t1 <- proc.time() - t1
cat(t1, "\n")

pred <- predict.XBCFdiscreteHeterosk3(fit, X_con = x, X_mod = x, Z = z, pihat = pihat, burnin = burnin)
tauhats <- pred$tau.adj.mean
muhats <- pred$mu.adj.mean

par(mfrow = c(3, 3))
plot(rowMeans(muhats) + mean(logt), mu, pch = 20, col = "darkgrey", main = "XBCF")
abline(0, 1, col = "red")


fit2 <- XBART.survival(
    y = matrix(logt) - mean(logt), X = x, delta = delta,
    num_sweeps = num_sweeps,
    burnin = burnin,
    p_categorical = 0,
    mtry = 10,
    num_trees_m = 35,
    max_depth_m = 15,
    Nmin_m = 1,
    num_cutpoints_m = 20,
    num_trees_v = 35,
    max_depth_v = 15,
    Nmin_v = 40,
    num_cutpoints_v = 20,
    ini_var = 0.5 * var(logt),
    ini_impute = ini_impute,
    verbose = FALSE,
    parallel = FALSE,
    sample_weights_flag = FALSE, a_scaling = TRUE, b_scaling = TRUE
)

pred2 <- predict(fit2, as.matrix(x))

plot(rowMeans(pred2$mhats) + mean(logt), mu, pch = 20, col = "darkgrey")
abline(0, 1, col = "red")

plot(rowMeans(sqrt(pred2$vhats)), sig, pch = 20, col = "darkgrey")
abline(0, 1, col = "red")



mu2 <- rowMeans(pred2$mhats)

rmst.xbart <- rep(0, ntest)

for (j in 1:num_sweeps) {
    for (i in 1:ntest) {
        rmst.xbart[i] <- rmst.xbart[i] + (1 / num_sweeps) * integrate(plnorm, 0, limit, meanlog = pred2$mhats[i, j] + mean(logt), sdlog = sqrt(pred2$vhats[i, j]), lower.tail = FALSE)$value
    }
}

plot(rmst.xbart, rmst.true, pch = 20, col = "black")
abline(0, 1, col = "red")




t.nft <- proc.time()
if (samp_nft) {
    x.train <- as.matrix(x)
    events <- seq(step, limit, step)
    nft.fit <- nft(x.train = x.train, times = Yobs, delta = delta, events = events, K = length(events), ndpost = 1000, nskip = 1000)
    pred <- predict(nft.fit, x.train[1:ntest, ])

    mat <- matrix(pred$surv.test.mean, ncol = length(events), byrow = TRUE)
    pred <- NULL

    sp.nft <- cbind(rep(1, ntest), mat[, -length(nft.fit$events)])
    timediff <- diff(c(0, nft.fit$events))
    rmst.nft <- rowSums(data.frame(mapply(`*`, as.data.frame(sp.nft), timediff, SIMPLIFY = FALSE)))
    gc()
}

# rmst.nft <- rmst.nft[1:ntest]
t.nft <- proc.time() - t.nft

points(rmst.nft, rmst.true, pch = 20, col = "red")
abline(0, 1, col = "red")

rmse.nft <- sqrt(mean((rmst.true - rmst.nft)^2))
rmse.x <- sqrt(mean((rmst.true - rmst.xbart)^2))


print(rmse.x)
print(rmse.nft)
