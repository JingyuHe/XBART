library(XBART)

set.seed(1324354676)
n = 10000


num_trees = 10
num_sweeps = 200
burnin = 20
parallel <- TRUE
nthread <- 4

x1 <- runif(n)
x2 <- runif(n, -3, 3)
z <- rnorm(n, 0, 2) + 2
# z <- rep(1, 100)
# y <- 10*x*(z==2)
y <- (sin(5 * x1) + x2) * z + cos(2 * x2 + x1)
x <- cbind(x1, x2)
# XBART::start_profiler("profiler.out")
time <- Sys.time()
fit <- XBART::XBCF_continuous(as.matrix(y), Z = as.matrix(z), X = as.matrix(x), X_ps = as.matrix(x), X_trt = as.matrix(x), Xtest = as.matrix(x), Xtest_ps = as.matrix(x), Xtest_trt = as.matrix(x), Ztest = as.matrix(z), parallel = parallel, num_trees = num_trees, num_trees_ps = num_trees, num_trees_trt = num_trees, num_sweeps = num_sweeps, burnin = burnin, nthread = nthread, sample_weights = TRUE)
time <- Sys.time() - time
print(time)
# XBART::stop_profiler()
pred <- rowMeans(fit$yhats_test)
inds1 <- z < 1.1 & z > 0.9
inds2 <- z > -1.1 & z < -0.9
par(mfrow = c(2,2))
plot(x1[inds1], pred[inds1], pch = 20, col = "green", xlim = c(min(x1), max(x1)))
points(x1[inds2], pred[inds2], pch = 20, col = "blue")
points(x1, (sin(5 * x1) + x2) * (1), col = "green")
points(x1, (sin(5 * x1) + x2) * (-1), col = "blue")
plot(y, pred)
abline(0, 1, col = "red")
plot(sin(5 * x1) + x2, pred / z, pch = 20)


# # XBART
# data = cbind(z, x)
# data <- as.matrix(data)
# time2 <- Sys.time()
# fit2 <- XBART(as.matrix(y), data, num_trees = num_trees, num_sweeps = num_sweeps, burnin = burnin, parallel = parallel, nthread = nthread)
# time2 = Sys.time() - time2
# print(time2)
