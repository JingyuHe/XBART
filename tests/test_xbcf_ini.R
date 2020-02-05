library(XBART)
library(bcf2)
library(dbarts)
# data generating process
n = 5000
# 
# set.seed(1)

confidence = 0.9

Coverage = list()
Length = list()
RMSE = list()

# parameters of XBCF
burnin = 25
sweeps = 100
treesmu = 60
treestau = 30


continuous = TRUE

# for (tt in 1:5) {
#     # loop over all replications, (generate different data)
#     cat("-------------------------\n")
#     cat("number of replications", tt, "\n")
#     cat("-------------------------\n")
    

    # data generating process
    x1 = rnorm(n)
    x2 = rbinom(n, 1, 0.2)
    x3 = sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
    # x2 = rnorm(n)
    # x3 = rnorm(n)
    
    x4 = rnorm(n)
    x5 = rbinom(n, 1, 0.7)
    # x5 = rnorm(n)
    x = cbind(x1, x2, x3, x4, x5)
    
    # alpha = 0.5
    tau = 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)
    
    ## RIC
    
    mu = function(x) {
        lev = c(-0.5, 0.75, 0)
        result = 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x3]
        
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
    
    t1 = proc.time()
    fit_xbcf = XBCF(y, x1, x, z, num_sweeps = sweeps, burnin = burnin, max_depth = 50, 
        Nmin = 1, num_cutpoints = 50, no_split_penality = "Auto", mtry_pr = ncol(x1), 
        mtry_trt = ncol(x), p_categorical_pr = 0, p_categorical_trt = 0, num_trees_pr = treesmu, 
        alpha_pr = 0.95, beta_pr = 1.25, tau_pr = tau1, kap_pr = 1, s_pr = 1, pr_scale = FALSE, 
        num_trees_trt = treestau, alpha_trt = 0.25, beta_trt = 2, tau_trt = tau2, 
        kap_trt = 1, s_trt = 1, trt_scale = TRUE, verbose = FALSE, a_scaling = TRUE, 
        b_scaling = TRUE)
    t1 = proc.time() - t1
    qhat_xbcf = rowSums(fit_xbcf$muhats[, (burnin + 1):sweeps])/(sweeps - burnin)

    # compute tauhats as (b1-b0)*tau
    th_xbcf = fit_xbcf$tauhats
    b_xbcf = fit_xbcf$b_draws
    seq <- (burnin + 1):sweeps
    for (i in seq) {
        th_xbcf[, i] = th_xbcf[, i] * (b_xbcf[i, 2] - b_xbcf[i, 1])
    }
    tauhats_xbcf = rowSums(th_xbcf[, (burnin + 1):sweeps])/(sweeps - burnin)
    # tauhats_xbcf = tauhats_xbcf * sdy
    plot(tau, tauhats_xbcf)
    abline(0, 1)

    mu_xbcf = rowMeans(fit_xbcf$muhats[, (burnin + 1):sweeps]) * sdy

    yhat_xbcf = rowMeans(fit_xbcf$muhats[, (burnin + 1):sweeps]) + tauhats_xbcf * z
    yhat_xbcf = yhat_xbcf * sdy

    # check bcf original
    t2 = proc.time()    


    # warm start
    t = proc.time() 
    fit_warmstart = bcf2::bcf_ini(as.vector(fit_xbcf$treedraws_pr[100]), as.vector(fit_xbcf$treedraws_trt[100]), fit_xbcf$a_draws[100, 1], fit_xbcf$b_draws[100, 1], fit_xbcf$b_draws[100, 2], fit_xbcf$sigma0_draws[1,100], y, z, x, x, pihat, nburn=0, nsim=100, include_pi = 'control',use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau) 
    t = proc.time() - t
    tau_post_warmstart = fit_warmstart$tau 
    that_warmstart = colMeans(tau_post_warmstart) 
    that_warmstart = that_warmstart*sdy 
    plot(tau, that_warmstart);
    abline(0,1)
    yhat_warmstart = colMeans(fit_warmstart$yhat) * sdy
    mu_warmstart = ( colMeans(fit_warmstart$yhat) - colMeans(fit_warmstart$tau) * z) * sdy

    fit_bcf = bcf::bcf(y, z, x, x, pihat, nburn=1000, nsim=1000, include_pi = 'control', use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)
    tau_post_bcf = fit_bcf$tau
    that_bcf = colMeans(tau_post_bcf)
    that_bcf = that_bcf * sdy
    yhat_bcf = colMeans(fit_bcf$yhat) * sdy
    mu_bcf = ( colMeans(fit_bcf$yhat) - colMeans(fit_bcf$tau) * z) * sdy
    
    fit_bcf2 = bcf2::bcf(y, z, x, x, pihat, nburn=1000, nsim=1000, include_pi = 'control', use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)
    tau_post_bcf2 = fit_bcf2$tau
    that_bcf2 = colMeans(tau_post_bcf2)
    that_bcf2 = that_bcf2 * sdy
    yhat_bcf2 = colMeans(fit_bcf2$yhat) * sdy
    mu_bcf2 = ( colMeans(fit_bcf2$yhat) - colMeans(fit_bcf2$tau) * z) * sdy


    RMSE_tau = c(sqrt(mean((tauhats_xbcf - tau)^2)), sqrt(mean((that_warmstart - tau)^2)), sqrt(mean((that_bcf - tau)^2)), sqrt(mean((that_bcf2 - tau)^2)))
    RMSE_Ey = c(sqrt(mean((Ey - yhat_xbcf)^2)), sqrt(mean((Ey - yhat_warmstart)^2)), sqrt(mean((Ey - yhat_bcf)^2)), sqrt(mean((Ey - yhat_bcf2)^2)))
    RMSE_mu = c(sqrt(mean((mu_xbcf - mu_true)^2)), sqrt(mean((mu_warmstart - mu_true)^2)), sqrt(mean((mu_bcf - mu_true)^2)), sqrt(mean((mu_bcf2 - mu_true)^2)))

    Result = rbind(RMSE_tau, RMSE_Ey, RMSE_mu)
    colnames(Result) = c("XBCF", "Warmstart + BCF2", "BCF", "BCF2")
    rownames(Result) = c("RMSE tau", "RMSE E(Y)", "RMSE mu")
    Result
    
    
#     ####################################################################### Calculate coverage
    
#     # coverage of the real average
#     draw_BCF_XBCF = c()
    
#     set.seed(1)
    
#     for (i in 51:100) {
#         # bart with XBART initialization
#         # discard first 50 trees
#         # initialize BCF at 50 trees, draw 100 samples for each
#         cat("------------- i ", i, "\n")
#         bcf_fit = bcf2::bcf_ini(as.vector(xbcf_fit$treedraws_pr[i]), as.vector(xbcf_fit$treedraws_trt[i]), 
#             xbcf_fit$a_draws[i, 1], xbcf_fit$b_draws[i, 1], xbcf_fit$b_draws[i, 2], 
#             xbcf_fit$sigma0_draws[1, i], y, z, x, x, pihat, nburn = 0, nsim = 100, 
#             include_pi = "control", use_tauscale = TRUE, ntree_control = treesmu, 
#             ntree_moderate = treestau)
        
#         draw_BCF_XBCF = rbind(draw_BCF_XBCF, bcf_fit$tau[21:100, ])
#     }
    
#     # scaling!
#     draw_BCF_XBCF = draw_BCF_XBCF * sdy

#     RMSE2 = sqrt(mean((colMeans(draw_BCF_XBCF) - tau)^2))
    
#     # run a full bcf chain
#     bcf_fit2 = bcf(y, z, x, x, pihat, nburn = 1000, nsim = 4000, include_pi = "control", 
#         use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)
#     tau_post2 = bcf_fit2$tau
#     that2 = colMeans(tau_post2)
#     that2 = that2 * sdy
    
#     RMSE3 = sqrt(mean((that2 - tau)^2))
    
    
#     coverage = c(0, 0, 0)
    
#     length = matrix(0, n, 3)
    
#     # scaling!
#     bcf_draw = bcf_fit2$tau * sdy
#     xbart_draw = th * sdy
    
#     for (i in 1:n) {
#         lower = quantile(xbart_draw[i, 15:50], (1 - confidence)/2)
#         higher = quantile(xbart_draw[i, 15:50], confidence + (1 - confidence)/2)
#         if (tau[i] < higher && tau[i] > lower) {
#             coverage[1] = coverage[1] + 1
#         }
#         length[i, 1] = higher - lower
        
#         lower = quantile(bcf_draw[, i], (1 - confidence)/2)
#         higher = quantile(bcf_draw[, i], confidence + (1 - confidence)/2)
#         if (tau[i] < higher && tau[i] > lower) {
#             coverage[2] = coverage[2] + 1
#         }
#         length[i, 2] = higher - lower
        
#         lower = quantile(draw_BCF_XBCF[, i], (1 - confidence)/2)
#         higher = quantile(draw_BCF_XBCF[, i], confidence + (1 - confidence)/2)
#         if (tau[i] < higher && tau[i] > lower) {
#             coverage[3] = coverage[3] + 1
#         }
#         length[i, 3] = higher - lower
        
#     }
    
#     coverage/n
#     colMeans(length)
    
    
#     Coverage[[tt]] = coverage
#     Length[[tt]] = length
#     RMSE[[tt]] = c(RMSE1, RMSE2, RMSE3)

#     cat("RMSE is ", RMSE[[tt]], "\n") 
# }


# # save(Length = Length, Coverage = Coverage, confidence = confidence,
# # file=paste(confidence, 'conf.rda', sep = ''))

# Length_ave = c(0, 0, 0)
# Coverage_ave = c(0, 0, 0)

# for (i in 1:length(Length)) {
#     Coverage_ave = Coverage_ave + as.vector(Coverage[[i]])
#     Length_ave = Length_ave + as.vector(colMeans(Length[[i]]))
# }

# Coverage_ave = Coverage_ave/length(Length)/5000
# Length_ave = Length_ave/length(Length)


# results = rbind(confidence, Coverage_ave, Length_ave)
# colnames(results) = c("XBCF", "BCF", "warm start")
# rownames(results) = c("confidence level", "Coverage", "Length")
# results

