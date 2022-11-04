# y = f(x) + s(x) Z
#basic parameters
set.seed(27)
n=500 #train (and test) sample sizes
p=1 #just one x

# train data
x = matrix(sort(runif(n*p)),ncol=p) #iid uniform x values
#x = matrix(sort(rbinom(n*p,1,0.5)),ncol=p)
fx = 4*(x[,1]^2) #quadratric function f
#fx = x[,1] #constant mean
sx = .8*exp(2*x[,1]) # exponential function s
#sx = .5*x[,1]
#sx = rep(.1,n)
y = fx + sx*rnorm(n)


# ##test data (the p added to the variable names is for predict)
np=1000
xp = matrix(sort(runif(np*p)),ncol=p)
fxp = 4*(xp[,1]^2)
#fxp = 0 #constant mean
sxp = .8*exp(2*xp[,1])
#sxp = .5*xp[,1]
yp = fxp + sxp*rnorm(n)

# #Now, let’s have a look at the simulated data:
# plot(x,y,ylab="y",cex.axis=1,cex.lab=1)
# lines(x,fx,col="blue",lwd=2)
# lines(x,fx+2*sx,col="green",lwd=2,lty=2)
# lines(x,fx-2*sx,col="green",lwd=2,lty=2)

### rbart
# first run at default settings
# resdef = rbart(x,y)
# # second run, this setting will give us a smoother function (k=5)
# # and use fewer iterations at each MCMC stage so that it runs faster
# res = rbart(x,y,nskip=100,ndpost=400,k=5,numcut=1000,nadapt=200,adaptevery=20,tc=5)
# resdefp = predict(resdef,x.test=xp) #get prediction for test x in xp, using resdef
# resp = predict(res,x.test=xp) 
# 
# plot(resp$mmean,yp,col="black",lty=1,lwd=2)


###################### xbart  #################
library(XBART)
num_sweeps = 80
burnin = 30

fit = XBART.heterosk(y=matrix(y),X=x, Xtest=xp, 
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
                     )
# predicted ys
y_hats <- rowMeans(fit$yhats_test[,burnin:num_sweeps])
plot(y_hats,yp,ylab="y",cex.axis=1,cex.lab=1)
abline(0,1)

# predicted sigma2
sig2_hats <- rowMeans(fit$sigma2hats_test[,burnin:num_sweeps])
plot(sig2_hats,sxp^2,ylab="var(y)",cex.axis=1,cex.lab=1)
abline(0,1)


# num_sweeps = 50
# burnin = 20
# fit.xb = XBART(y=matrix(y),X=x, Xtest=x,
#              num_sweeps = num_sweeps,
#              burnin = burnin,
#              mtry = 1,
#              p_categorical = 0,
#              num_trees = 20,
#              max_depth = 250,
#              Nmin = 1,
#              num_cutpoints = 20,
#              verbose = TRUE,
#              parallel = FALSE,
#   )
# 
# y_hats <- rowMeans(fit.xb$yhats_test[,burnin:num_sweeps])
# plot(y_hats,y,ylab="y",cex.axis=1,cex.lab=1)
# abline(0,1)
