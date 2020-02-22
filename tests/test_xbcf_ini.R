rm(list = ls())
library(XBART)
library(bcf2)
library(dbarts)
# data generating process
n = 5000

seed = 1
set.seed(seed)

# parameters of XBCF
burnin = 25
sweeps = 100
treesmu = 10
treestau = 10

# DGP continuous
continuous = TRUE


if(continuous){
    p_categorical_pr = 0
    p_categorical_trt = 0
}else{
    p_categorical_pr = 5
    p_categorical_trt = 5
}


if(continuous){
    x1 = rnorm(n)
    x2 = rnorm(n)
    x3 = rnorm(n)
    x4 = rnorm(n)
    x5 = rnorm(n)
    x = cbind(x1, x2, x3, x4, x5)
    
    # alpha = 0.5
    tau = 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)
    
    ## RIC
    
    mu = function(x) {
        lev = c(-0.5, 0.75, 0)
        result = 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) # + lev[x3]
        # nonlinear result = 1 + abs(x[,1])*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
        return(result)
    }
    
    pi = pnorm(-0.5 + mu(x) - x[, 2] + 0 * x[, 4], 0, 3)
    hist(pi, 100)
    z = rbinom(n, 1, pi)
    
    Ey = mu(x) + tau * z

    mu_true = mu(x)
    
    sig = 0.25 * sd(Ey)
    
    y = Ey + sig * rnorm(n)
    # If you didn't know pi, you would estimate it here
    pihat = pi
    
    y = y - mean(y)
    sdy = sd(y)
    y = y/sdy
    tau1 = 0.9 * var(y)/treesmu
    tau2 = 0.1 * var(y)/treestau
    x <- data.frame(x)
    x1 <- cbind(pihat, x)
    x = as.matrix(x)
    x1 = as.matrix(x1)

}else{

    x1 = rnorm(n)
    x2 = rbinom(n, 1, 0.2)
    x3 = sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
    x4 = rnorm(n)
    x5 = rbinom(n, 1, 0.7)
    x = cbind(x1, x2, x3, x4, x5)
    
    # alpha = 0.5
    tau = 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)
    
    ## RIC
    
    mu = function(x) {
        lev = c(-0.5, 0.75, 0)
        result = 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2]))  + lev[x3]
        # nonlinear result = 1 + abs(x[,1])*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
        return(result)
    }
    
    pi = pnorm(-0.5 + mu(x) - x[, 2] + 0 * x[, 4], 0, 3)
    hist(pi, 100)
    z = rbinom(n, 1, pi)
    
    Ey = mu(x) + tau * z

    mu_true = mu(x)
    
    sig = 0.25 * sd(Ey)
    
    y = Ey + sig * rnorm(n)
    # If you didn't know pi, you would estimate it here
    pihat = pi
    
    y = y - mean(y)
    sdy = sd(y)
    y = y/sdy
    tau1 = 0.9 * var(y)/treesmu
    tau2 = 0.1 * var(y)/treestau
    x <- data.frame(x)
    x[, 3] <- as.factor(x[, 3])
    x <- makeModelMatrixFromDataFrame(data.frame(x))
    x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])
    x1 <- cbind(pihat, x)
    x = as.matrix(x)
    x1 = as.matrix(x1)
}



fit_xbcf = XBCF(y, x1, x, z, num_sweeps = sweeps, burnin = burnin, max_depth = 50, 
    Nmin = 1, num_cutpoints = 50, no_split_penality = "Auto", mtry_pr = ncol(x1), 
    mtry_trt = ncol(x), p_categorical_pr = p_categorical_pr, p_categorical_trt = p_categorical_trt, num_trees_pr = treesmu, 
    alpha_pr = 0.95, beta_pr = 1.25, tau_pr = tau1, kap_pr = 1, s_pr = 1, pr_scale = FALSE, 
    num_trees_trt = treestau, alpha_trt = 0.25, beta_trt = 2, tau_trt = tau2, 
    kap_trt = 1, s_trt = 1, trt_scale = TRUE, verbose = FALSE, a_scaling = TRUE, 
    b_scaling = TRUE, random_seed = seed)
qhat_xbcf = rowSums(fit_xbcf$muhats[, (burnin + 1):sweeps])/(sweeps - burnin)

# compute tauhats as (b1-b0)*tau
th_xbcf = fit_xbcf$tauhats
b_xbcf = fit_xbcf$b_draws
seq <- (burnin + 1):sweeps
for (i in seq) {
    th_xbcf[, i] = th_xbcf[, i] * (b_xbcf[i, 2] - b_xbcf[i, 1])
}
tauhats_xbcf = rowSums(th_xbcf[, (burnin + 1):sweeps])/(sweeps - burnin)
tauhats_xbcf = tauhats_xbcf * sdy
plot(tau, tauhats_xbcf)
abline(0, 1)
a_xbcf = fit_xbcf$a_draws
mu_xbcf = fit_xbcf$muhats
for (i in seq) {
    mu_xbcf[, i] = mu_xbcf[, i] * a_xbcf[i]
}
mu_xbcf = rowMeans(mu_xbcf[, (burnin + 1):sweeps]) * sdy
yhat_xbcf = mu_xbcf + tauhats_xbcf * z

 
# # BCF package
# fit_bcf = bcf::bcf(y, z, x, x, pihat, nburn=0, nsim=100, include_pi = 'control', use_tauscale = TRUE, use_muscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)
# tau_post_bcf = fit_bcf$tau
# that_bcf = colMeans(tau_post_bcf)
# that_bcf = that_bcf * sdy
# yhat_bcf = colMeans(fit_bcf$yhat) * sdy
# mu_bcf = (colMeans(fit_bcf$yhat) - colMeans(fit_bcf$tau) * z) * sdy



# # BCF2 package
fit_bcf2 = bcf2::bcf(y, z, x, x, pihat, nburn=100, nsim=2, include_pi = 'control', use_tauscale = TRUE, use_muscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau) 
tau_post_bcf2 = fit_bcf2$tau
that_bcf2 = colMeans(tau_post_bcf2)
that_bcf2 = that_bcf2 * sdy
yhat_bcf2 = colMeans(fit_bcf2$yhat) * sdy
mu_bcf2 = (colMeans(fit_bcf2$yhat) - colMeans(fit_bcf2$tau) * z) * sdy



# # compare estimations from BCF2 and XBCF
# est = c(fit_xbcf$a_draws[100, 1], fit_xbcf$b_draws[100, 1], fit_xbcf$b_draws[100, 2], fit_bcf2$mscale, fit_bcf2$bscale0, fit_bcf2$bscale1)
# est = matrix(est, 2, 3, byrow = TRUE)
# colnames(est) = c("mscale", "bscale0", "bscale1")
# rownames(est) = c("XBCF", "BCF2")
# print(est)



# initialize BCF2 at XBART
n_draw_warmstart = 2
burnin_warmstart = 0

# pi_con_sigma_ini = abs(fit_xbcf$sigma0_draws[1,100] / fit_xbcf$a_draws[100, 1])
# pi_mod_sigma_ini = abs(fit_xbcf$sigma0_draws[1,100])
pi_con_sigma_ini = fit_bcf2$pi_con_sigma
pi_mod_sigma_ini = fit_bcf2$pi_mod_sigma

if(0){
    b0_ini = -0.5
    b1_ini = 0.5
    # this is b1 - b0, used to scale tau(x)
    mod_tree_scaling = 1.0 / (fit_xbcf$b_draws[100, 2] - fit_xbcf$b_draws[100, 1])
}else{
    b0_ini = fit_xbcf$b_draws[100, 1]
    b1_ini = fit_xbcf$b_draws[100, 2]
    mod_tree_scaling = 1
}

fit_warmstart = bcf2::bcf_ini(as.vector(fit_xbcf$treedraws_pr[100]), as.vector(fit_xbcf$treedraws_trt[100]), fit_xbcf$a_draws[100, 1], b0_ini, b1_ini, mod_tree_scaling = mod_tree_scaling, fit_xbcf$sigma0_draws[1,100], fit_bcf2$pi_con_tau, pi_con_sigma_ini, fit_bcf2$pi_mod_tau, pi_mod_sigma_ini, y, z, x, x, pihat, nburn=0, nsim=n_draw_warmstart, include_pi = 'control',use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau, ini_bcf = FALSE) 

# fit_warmstart = bcf2::bcf_ini(as.vector(fit_xbcf$treedraws_pr[100]), as.vector(fit_xbcf$treedraws_trt[100]), fit_xbcf$a_draws[100, 1], fit_xbcf$b_draws[100, 1], fit_xbcf$b_draws[100, 2], fit_xbcf$sigma0_draws[1,100], fit_bcf2$pi_con_tau, fit_bcf2$pi_con_sigma, fit_bcf2$pi_mod_tau, fit_bcf2$pi_mod_sigma, y, z, x, x, pihat, nburn=burnin_warmstart, nsim=n_draw_warmstart, include_pi = 'control',use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau, ini_bcf = FALSE) 


# Nikolay's 
# fit_warmstart = bcf2::bcf_ini(as.vector(fit_xbcf$treedraws_pr[100]), as.vector(fit_xbcf$treedraws_trt[100]), fit_xbcf$a_draws[100, 1], fit_xbcf$b_draws[100, 1], fit_xbcf$b_draws[100, 2], fit_xbcf$sigma0_draws[1,100], fit_bcf2$pi_con_tau, fit_bcf2$pi_con_sigma, fit_bcf2$pi_mod_tau, fit_bcf2$pi_mod_sigma,  y, z, x, x, pihat, nburn=0, nsim=100, include_pi = 'control',use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau, ini_bcf = FALSE) 


# fit_warmstart = bcf2::bcf_ini(fit_bcf2$tree_con, fit_bcf2$tree_mod, fit_xbcf$a_draws[100, 1], fit_xbcf$b_draws[100, 1], fit_xbcf$b_draws[100, 2], fit_xbcf$sigma0_draws[1,100], fit_bcf2$pi_con_tau, fit_bcf2$pi_con_sigma, fit_bcf2$pi_mod_tau, fit_bcf2$pi_mod_sigma, y, z, x, x, pihat, nburn=0, nsim=100, include_pi = 'control',use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau, ini_bcf = FALSE) 


# # initialize BCF2 at BCF2, for debugging purpose
# fit_warmstart = bcf2::bcf_ini(treedraws_con = fit_bcf2$tree_con, treedraws_mod = fit_bcf2$tree_mod, muscale_ini = fit_bcf2$mscale, bscale0_ini = fit_bcf2$bscale0, bscale1_ini = fit_bcf2$bscale1, sigma_ini = fit_bcf2$sigma[length(fit_bcf2$sigma)], pi_con_tau = fit_bcf2$pi_con_tau, pi_con_sigma = fit_bcf2$pi_con_sigma, pi_mod_tau = fit_bcf2$pi_mod_tau, pi_mod_sigma = fit_bcf2$pi_mod_sigma, mod_tree_scaling = 1, y, z, x, x, pihat, nburn=burnin_warmstart, nsim=n_draw_warmstart, ntree_control = treesmu, ntree_moderate = treestau, include_pi = 'control',use_tauscale = TRUE, ini_bcf = TRUE)



# # some strange mixture of BCF2 and XBART initialization
# fit_warmstart = bcf2::bcf_ini(as.vector(fit_xbcf$treedraws_pr[100]), fit_xbcf$treedraws_trt[100], fit_xbcf$a_draws[100, 1], fit_bcf2$bscale0, fit_bcf2$bscale1, fit_bcf2$sigma[length(fit_bcf2$sigma)], fit_bcf2$pi_con_tau, fit_bcf2$pi_con_sigma, fit_bcf2$pi_mod_tau, fit_bcf2$pi_mod_sigma, y, z, x, x, pihat, nburn=0, nsim=100, include_pi = 'control',use_tauscale = FALSE, use_muscale = FALSE, ntree_control = treesmu, ntree_moderate = treestau, ini_bcf = FALSE) 




tau_post_warmstart = matrix(fit_warmstart$tau, n_draw_warmstart, n)
that_warmstart = colMeans(tau_post_warmstart) 

that_warmstart = that_warmstart*sdy 
plot(tau, that_warmstart);
abline(0,1)
yhat_warmstart = colMeans(matrix(fit_warmstart$yhat, n_draw_warmstart, n)) * sdy
mu_warmstart = ( colMeans(matrix(fit_warmstart$yhat, n_draw_warmstart, n)) - colMeans(matrix(fit_warmstart$tau, n_draw_warmstart, n)) * z ) * sdy


par(mfrow = c(2,2))
plot(tau, tauhats_xbcf, main = "XBCF")
abline(0,1, col = "red", lwd = 2)
plot(tau, that_bcf, main = "BCF")
abline(0,1, col = "red", lwd = 2)
plot(tau, that_bcf2, main = "BCF2")
abline(0,1, col = "red", lwd = 2)
plot(tau, that_warmstart, main = "warmstart")
abline(0,1, col = "red", lwd = 2)


RMSE_tau = c(sqrt(mean((tauhats_xbcf - tau)^2)), sqrt(mean((that_warmstart - tau)^2)), sqrt(mean((that_bcf - tau)^2)), sqrt(mean((that_bcf2 - tau)^2)))
RMSE_Ey = c(sqrt(mean((Ey - yhat_xbcf)^2)), sqrt(mean((Ey - yhat_warmstart)^2)), sqrt(mean((Ey - yhat_bcf)^2)), sqrt(mean((Ey - yhat_bcf2)^2)))
RMSE_mu = c(sqrt(mean((mu_xbcf - mu_true)^2)), sqrt(mean((mu_warmstart - mu_true)^2)), sqrt(mean((mu_bcf - mu_true)^2)), sqrt(mean((mu_bcf2 - mu_true)^2)))

Result = rbind(RMSE_tau, RMSE_Ey, RMSE_mu)
colnames(Result) = c("XBCF", "Warmstart + BCF2", "BCF", "BCF2")
rownames(Result) = c("RMSE tau", "RMSE E(Y)", "RMSE mu")
Result

