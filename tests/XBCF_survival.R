library(XBART)
library(nftbart)

# set.seed(11234)

samp_nft <- TRUE
ini_impute <- -0.1

cen <- c(20, 50, 80)
cf <- c(3.55, 3, 2.45)

opt <- 2

#####################################################
### GENERATE DATA ###
n <- 1000
x1 <- 2 * rnorm(n)
x2 <- runif(n, 1, 2)
x3 <- runif(n, 0, 1)
x4 <- runif(n, 0, 1)

x <- data.frame(x1, x2, x3, x4)

mu <- 1.5 + 0.5 * x[, 2] + 0.5 * x[, 4]


# define treatment effects
tau <- 3

# compute propensity scores and treatment assignment
# pi <- pnorm(-0.5 + mu - x[, 2] + 0. * x[, 4], 0, 3)

pi <- rep(1 / 2, n)

z <- rbinom(n, 1, pi)

Ey <- mu + 1 * tau * z

# Ey  =mu

sig <- (1 + 1 * abs(x[, 1]))

Y <- rnorm(n, Ey, sig)

# C <- rlnorm(n, cf[opt] - 0.1 * x[, 2], 0.5 + 0.05 * abs(x[, 1]))
C <- rnorm(n, 3.3, 1)
# hist(C,30)

Yobs <- pmin(Y, C)
delta <- as.numeric(Y < C)

#  Yobs = Y
#  delta = rep(1, n)


# set time limit (based on Yobs distribution), step for tgrid
limit <- quantile(exp(Yobs), 0.9)
step <- 0.25
tgrid <- seq(0, limit, step)


ntest <- 200
rmst.true <- rep(0, ntest)
for (i in 1:ntest) {
    rmst.true[i] <- integrate(plnorm, 0, limit, meanlog = Ey[i], sdlog = sig[i], lower.tail = FALSE, subdivisions = 1000)$value
}



# convert to log scale

logt <- (Yobs)

#####################################################
# XBCF survival, heteroskedastic, treatment modified variance
num_sweeps <- 500
burnin <- 200
mlogt <- mean(logt)
sdlogt <- sd(logt)
yscale <- (logt - mlogt) / sdlogt

pihat <- pi
a_scaling <- TRUE
b_scaling <- TRUE


t1 = proc.time()
fit <- XBCF.survival.discrete.heterosk3(
    y = yscale, Z = z, delta = delta, X_con = x, X_mod = x,
    pihat = pihat, p_categorical_con = 0, p_categorical_mod = 0,
    num_trees_con = 100, num_trees_mod = 20, num_trees_v = 5,
    num_sweeps = num_sweeps, ini_var = 1, ini_impute = ini_impute,
    burnin = burnin, sample_weights = TRUE,
    a_v_con = 5, b_v_con = 5, a_v_mod = 5, b_v_mod = 5,
    a_scaling = a_scaling, b_scaling = b_scaling, verbose = FALSE
)

t1 <- proc.time() - t1
cat(t1, "\n")

pred <- predict.XBCFdiscreteHeterosk3(fit, X_con = x, X_mod = x, Z = z, pihat = pihat, burnin = burnin)
tauhats <- pred$tau.adj.mean
muhats <- pred$mu.adj.mean

par(mfrow = c(3, 3))
plot(sdlogt * (pred$yhats.adj.mean) + mlogt, Ey, pch = 20, col = z + 1, main = "yhats, XBCF survival")
abline(0, 1, col = "red")

plot(sdlogt * rowMeans(sqrt(pred$variance)), sig, pch = 20, col = z + 1, main = "std, XBCF survival")
abline(0, 1, col = "red")

rmst.xbcf = rep(0, ntest)

for (j in (burnin + 1):num_sweeps) {
    for (i in 1:ntest) {
        rmst.xbcf[i] <- rmst.xbcf[i] + (1 / (num_sweeps - burnin)) * integrate(plnorm, 0, limit, meanlog = sdlogt * (pred$yhats.adj[i, j]) + mlogt, sdlog = sdlogt * sqrt(pred$variance[i, j]), lower.tail = FALSE, subdivisions = 1000)$value
    }
}

plot(rmst.xbcf, rmst.true, pch = 20, col = z + 1)
abline(0, 1, col = "red")

rmse.xbcf <- sqrt(mean((rmst.true - rmst.xbcf)^2))


#######
# XBCF
# run XBCF (XBART repo)
t2 <- proc.time()
xbcf.fit.xb <- XBCF.discrete(y = yscale, Z = z, X_con = x, X_mod = x, pihat = pihat, p_categorical_con = 1, p_categorical_mod = 1, num_sweeps = num_sweeps, burnin = burnin, num_trees_con = 100, num_trees_mod = 20)
t2 <- proc.time() - t2

pred2 <- predict(xbcf.fit.xb, X_con = x, X_mod = x, Z = z, pihat = pihat, burnin = 30)

# taus2 <- pred$tau.adj
# tauhats2 = rowMeans(taus2)
tauhats2 <- pred$tau.adj.mean
muhats2 <- pred$mu.adj.mean

plot(sdlogt * (pred2$yhats.adj.mean) + mlogt, Ey, pch = 20, col = z + 1, main = "yhats, XBCF plain")
abline(0, 1, col = "red")


#####################################################
# XBART survival, homoskedastic
x.xbart <- cbind(x, z)
fit2 <- XBART.survival(
    y = yscale, X = x.xbart, delta = delta,
    num_sweeps = num_sweeps,
    burnin = burnin,
    p_categorical = 1,
    mtry = 5,
    num_trees_m = 100,
    max_depth_m = 20,
    Nmin_m = 2,
    num_cutpoints_m = 20,
    num_trees_v = 20,
    max_depth_v = 20,
    Nmin_v = 5,
    sampling_tau = TRUE,
    num_cutpoints_v = 20,
    ini_var = 1,
    ini_impute = ini_impute,
    verbose = FALSE,
    parallel = FALSE,
    sample_weights_flag = FALSE
)

pred2 <- predict(fit2, as.matrix(x.xbart))

plot(sdlogt * rowMeans(pred2$mhats) + mlogt, Ey, pch = 20, col = z + 1, main = "yhats, XBART")
abline(0, 1, col = "red")

plot(sdlogt * rowMeans(sqrt(pred2$vhats)), sig, pch = 20, col = z + 1, main = "std, XBART")
abline(0, 1, col = "red")


rmst.xbart <- rep(0, ntest)

for (j in (burnin + 1):num_sweeps) {
    for (i in 1:ntest) {
        rmst.xbart[i] <- rmst.xbart[i] + (1 / (num_sweeps - burnin)) * integrate(plnorm, 0, limit, meanlog = sdlogt * pred2$mhats[i, j] + mlogt, sdlog = sdlogt * sqrt(pred2$vhats[i, j]), lower.tail = FALSE, subdivisions = 1000)$value
    }
}

plot(rmst.xbart, rmst.true, pch = 20, col = z + 1)
abline(0, 1, col = "red")



# #####################################################
# # nft BART
# t.nft <- proc.time()
# if (samp_nft) {
#     x.train <- as.matrix(x)
#     events <- seq(step, limit, step)
#     nft.fit <- nft(x.train = x.train, times = Yobs, delta = delta, events = events, K = length(events), ndpost = 1000, nskip = 1000)
#     pred <- predict(nft.fit, x.train[1:ntest, ])

#     mat <- matrix(pred$surv.test.mean, ncol = length(events), byrow = TRUE)
#     # pred <- NULL

#     sp.nft <- cbind(rep(1, ntest), mat[, -length(nft.fit$events)])
#     timediff <- diff(c(0, nft.fit$events))
#     rmst.nft <- rowSums(data.frame(mapply(`*`, as.data.frame(sp.nft), timediff, SIMPLIFY = FALSE)))
#     gc()
# } else {
#     rmst.nft <- rep(0, ntest)
# }

# # rmst.nft <- rmst.nft[1:ntest]
# t.nft <- proc.time() - t.nft

# points(rmst.nft, rmst.true, pch = 20, col = "red")
# abline(0, 1, col = "red")


#####################################################
# results
rmse.xbcf <- sqrt(mean((rmst.true - rmst.xbcf)^2))
# rmse.nft <- sqrt(mean((rmst.true - rmst.nft)^2))
rmse.nft <- 0
rmse.xbart <- sqrt(mean((rmst.true - rmst.xbart)^2))

results <- c(rmse.xbcf, rmse.nft, rmse.xbart)
names(results) <- c("XBCF", "NFT", "XBART survival")
print("RMSE of all methods \n")
print(results)

plot(fit$a,main = "trace of a, XBCF survival")
