library(XBART)

set.seed(1324354676)

x <- sample(seq(1, 10, by = 0.01), size = 100, replace = T)

z <- sample(c(0, 1, 2), size = 100, replace = T)

y <- sin(x) * z

x <- as.matrix(x)

time <- Sys.time()
fit <- XBCF_continuous(as.matrix(y), Z = as.matrix(z), X = as.matrix(x), Xtest = as.matrix(x), Ztest = as.matrix(z), num_trees = 10, num_sweeps = 2000, burnin = 1000)
Sys.time() - time

pred <- rowMeans(fit$yhats_test)

plot(x, y)
points(x, pred, col = "red")
points(x[z == 1], pred[z == 1], col = "green")
points(x[z == 2], pred[z == 2], col = "blue")
curve(sin(x), add = T, col = "green")
curve(sin(x) * 2, add = T, col = "blue")

plot(y, pred)
abline(0, 1, col = "red")
