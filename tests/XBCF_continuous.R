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
x4 <- rbinom(n, 1, 0.2)
x5 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))

tret = (sin(5 * x1) + x2)

# different X matrix for prognostic and treatment trees
x_con <- cbind(x, x3, x4, x5)
x_mod <- cbind(x, x4, x5)

time <- Sys.time()
fit <- XBART::XBCF.continuous(as.matrix(y), Z = as.matrix(z), X_con = as.matrix(x_con), X_mod = as.matrix(x_mod), parallel = parallel, num_trees_con = 20, num_trees_mod = 10, mtry_con = 2, mtry_mod = 2, num_sweeps = num_sweeps, burnin = burnin, nthread = nthread, sample_weights = TRUE, verbose = FALSE, p_categorical_con = 2, p_categorical_mod = 2)
time <- Sys.time() - time
print(time)

# predict function return three terms
# mu, tau and yhats
# yhats = mu + z * tau
pred <- predict(fit, X_con = as.matrix(x_con), X_mod = as.matrix(x_mod), Z = as.matrix(z))
tau_hat = pred$tau
pred <- rowMeans(pred$yhats)

print(mean((tau_hat - tret)^2))

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
