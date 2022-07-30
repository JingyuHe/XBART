library(XBART)

set.seed(1324354676)
n <- 10000


num_trees <- 50
num_sweeps <- 200
burnin <- 20
parallel <- TRUE
nthread <- 4

x1 <- runif(n)
x2 <- runif(n, -3, 3)
z <- rnorm(n, 0, 2) + 2
# z <- rep(1, 100)
# y <- 10*x*(z==2)
y <- (sin(5 * x1) + x2) * z + cos(2 * x2 + x1)
x <- cbind(x1, x2)
x3 <- rnorm(n)

# different X matrix for prognostic and treatment trees
x_ps <- cbind(x, x3)
x_trt <- x

time <- Sys.time()
fit <- XBART::XBCF.continuous(as.matrix(y), Z = as.matrix(z), X_ps = as.matrix(x_ps), X_trt = as.matrix(x_trt), parallel = parallel, num_trees_ps = 20, num_trees_trt = 30, mtry_ps = 2, mtry_trt = 2, num_sweeps = num_sweeps, burnin = burnin, nthread = nthread, sample_weights = TRUE)
time <- Sys.time() - time
print(time)

# predict function return three terms
# mu, tau and yhats
# yhats = mu + z * tau
pred <- predict(fit, X_ps = as.matrix(x_ps), X_trt = as.matrix(x_trt), Z = as.matrix(z))

pred <- rowMeans(pred$yhats)

inds1 <- z < 1.1 & z > 0.9
inds2 <- z > -1.1 & z < -0.9
par(mfrow = c(2, 2))
plot(x1[inds1], pred[inds1], pch = 20, col = "green", xlim = c(min(x1), max(x1)))
points(x1[inds2], pred[inds2], pch = 20, col = "blue")
points(x1, (sin(5 * x1) + x2) * (1), col = "green")
points(x1, (sin(5 * x1) + x2) * (-1), col = "blue")
plot(y, pred)
abline(0, 1, col = "red")
plot(sin(5 * x1) + x2, pred / z, pch = 20)


# XBART
data <- cbind(z, x)
data <- as.matrix(data)
time2 <- Sys.time()
fit2 <- XBART(as.matrix(y), data, num_trees = num_trees, num_sweeps = num_sweeps, burnin = burnin, parallel = parallel, nthread = nthread)
time2 <- Sys.time() - time2
print(time2)
