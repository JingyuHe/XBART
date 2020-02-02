library(XBART)
library(bcf2)
library(dbarts)
# data generating process
n = 5000
#
set.seed(1)

confidence = 0.80

Coverage = list()
Length = list()

for(tt in 1:50){


x1 = rnorm(n)
x2 = rbinom(n,1,0.2)
x3 = sample(1:3,n,replace=TRUE,prob = c(0.1,0.6,0.3))

x4 = rnorm(n)
x5 = rbinom(n,1,0.7)
x = cbind(x1,x2,x3,x4,x5)

#alpha = 0.5
tau = 2 + 0.5*x[,4]*(2*x[,5]-1)



## RIC

mu = function(x){
  lev = c(-0.5,0.75,0)
  result = 1 + x[,1]*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
  
  # nonlinear
  #result = 1 + abs(x[,1])*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
  
  return(result)
}

pi = pnorm(-0.5 + mu(x) - x[,2] + 0.*x[,4],0,3)
hist(pi,100)
z = rbinom(n,1,pi)

Ey = mu(x) + tau*z

sig = 0.25*sd(Ey)

y = Ey + sig*rnorm(n)
# If you didn't know pi, you would estimate it here
pihat = pi

y = y - mean(y)
sdy = sd(y)
y = y/sdy

burnin = 25
sweeps = 100
treesmu = 60
treestau = 30

tau1 = 0.9*var(y)/treesmu
tau2 = 0.1*var(y)/treestau

x <- data.frame(x)
x[,3] <- as.factor(x[,3])
x <- makeModelMatrixFromDataFrame(data.frame(x))
x <- cbind(x[,1],x[,6],x[,-c(1,6)])
x1 <- cbind(pihat,x)

t1 = proc.time()
xbcf_fit = XBCF(y, x1, x, z, num_sweeps = sweeps, burnin = burnin, max_depth = 50, Nmin = 1, num_cutpoints = 30, no_split_penality = "Auto",
                mtry_pr = ncol(x1), mtry_trt = ncol(x), p_categorical_pr = 5,  p_categorical_trt = 5,
                num_trees_pr = treesmu, alpha_pr = 0.95, beta_pr = 1.25, tau_pr = tau1, kap_pr = 1, s_pr = 1, pr_scale = FALSE,
                num_trees_trt = treestau, alpha_trt = 0.25, beta_trt = 2, tau_trt = tau2, kap_trt =1, s_trt = 1, trt_scale = FALSE, verbose = FALSE, a_scaling = TRUE, b_scaling = TRUE)
t1 = proc.time() - t1


qhat = rowSums(xbcf_fit$muhats[,(burnin+1):sweeps])/(sweeps-burnin)

# compute tauhats as (b1-b0)*tau
th = xbcf_fit$tauhats
b = xbcf_fit$b_draws
seq <- (burnin+1):sweeps
for (i in seq)
{ th[,i] = th[,i] * (b[i,2] - b[i,1]) }
tauhats = rowSums(th[,(burnin+1):sweeps])/(sweeps-burnin)
tauhats = tauhats*sdy
plot(tau, tauhats); abline(0,1)

# check bcf original
t2 = proc.time()


# # print("con trees \n")
# # print(as.vector(xbcf_fit$treedraws_pr[100]))

# # print("mod trees \n")
# # print(as.vector(xbcf_fit$treedraws_trt[100]))

# t = proc.time()
# bcf_fit = bcf2::bcf_ini(as.vector(xbcf_fit$treedraws_pr[100]), as.vector(xbcf_fit$treedraws_trt[100]), xbcf_fit$a_draws[100, 1], xbcf_fit$b_draws[100, 1], xbcf_fit$b_draws[100, 2], xbcf_fit$sigma0_draws[1,100], y, z, x, x, pihat, nburn=0, nsim=500, include_pi = "control",use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)
# t = proc.time() - t

# bcf_fit2 = bcf(y, z, x, x, pihat, nburn=0, nsim=500, include_pi = "control", use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)

# t2 = proc.time() - t2
# # Get posterior of treatment effects
# tau_post = bcf_fit$tau
# that = colMeans(tau_post)
# that = that*sdy
# plot(tau, that); abline(0,1)


# print(sqrt(mean((tauhats - tau)^2)))
# print(sqrt(mean((that - tau)^2)))






#######################################################################
# Calculate coverage
#######################################################################

# coverage of the real average
draw_BCF_XBCF = c()

set.seed(1)

for(i in 51:100){
  # bart with XBART initialization
  cat("------------- i ", i , "\n")
  bcf_fit = bcf2::bcf_ini(as.vector(xbcf_fit$treedraws_pr[i]), as.vector(xbcf_fit$treedraws_trt[i]), xbcf_fit$a_draws[i, 1], xbcf_fit$b_draws[i, 1], xbcf_fit$b_draws[i, 2], xbcf_fit$sigma0_draws[1,i], y, z, x, x, pihat, nburn=0, nsim=100, include_pi = "control",use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)

  draw_BCF_XBCF = rbind(draw_BCF_XBCF, bcf_fit$tau[21:100,])
}

draw_BCF_XBCF = draw_BCF_XBCF * sdy


bcf_fit2 = bcf(y, z, x, x, pihat, nburn=1000, nsim=4000, include_pi = "control", use_tauscale = TRUE, ntree_control = treesmu, ntree_moderate = treestau)


coverage = c(0,0,0)

length = matrix(0, n, 3)

bcf_draw = bcf_fit2$tau * sdy
xbart_draw = th * sdy

for(i in 1:n){
  lower = quantile(xbart_draw[i, 15:50], (1 - confidence) / 2)
  higher = quantile(xbart_draw[i, 15:50], confidence + (1 - confidence) / 2)
  if(tau[i] < higher && tau[i] > lower){
    coverage[1] = coverage[1] + 1
  }
  length[i,1] = higher - lower

  lower = quantile(bcf_draw[,i], (1 - confidence) / 2)
  higher = quantile(bcf_draw[,i], confidence + (1 - confidence) / 2)
  if(tau[i] < higher && tau[i] > lower){
    coverage[2] = coverage[2] + 1
  }
  length[i,2] = higher - lower

  lower = quantile(draw_BCF_XBCF[,i], (1 - confidence) / 2)
  higher = quantile(draw_BCF_XBCF[,i], confidence + (1 - confidence) / 2)
  if(tau[i] < higher && tau[i] > lower){
    coverage[3] = coverage[3] + 1
  }
  length[i,3] = higher - lower

}

coverage / n
colMeans(length)


Coverage[[tt]] = coverage
Length[[tt]] = length
}



save(Length = Length, Coverage = Coverage, confidence = confidence, file=paste(confidence, "conf.rda", sep = ""))

