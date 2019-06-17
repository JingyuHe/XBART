library("XBART")

printTPR<- function(pihat,ytrue,add=FALSE,col = 'steelblue'){
    
    thresh <- sort(pihat)
    N <- length(pihat)
    yhat <- sapply(1:N, function(a) as.double(pihat >= thresh[a]))
    tpr <- sapply(1:N, function(a) length(which(ytrue==1 & yhat[,a]==1))/sum(ytrue==1))
    print(mean(tpr))
}

# Data Generating Function
generate_function <- function(number=0){
  if(number ==0){return (function(x) {beta = seq(-2,2,length.out=d);return(1 + x%*%beta);} )}# Linear
  else if(number == 1) {return(function(x){sin((x[,2]-2*x[,3] + x[,1]))})  } # Single Index
  else if(number == 2) {return(function(x){5*sin(3*x[,1])+2*x[,2]^2 + 3*(x[,3]*x[,4])}) } #Friedman
  else if(number == 3) {return(function(x){
    avec = seq(-1.5,1.5,length.out = 10)
    ind = rowSums((x[,1:10]-matrix(avec,dim(x)[1],10,byrow=TRUE))^2)
    10*sqrt(ind) + sin(5*sqrt(ind))}  
  ) } # Stepfunction
  else{return (function(x){ 1 + 3*apply(x[,1:3],1,max)})} # Max
  #else {return(function(x){10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]}) }
} 


#set.seed(7215)
n = 10000
nt = 2500
#nt = 1000
a = 5

dcat = 0


d = 30
k = d
x = matrix(runif(n*d,-a,a),n,d)
xtest = matrix(runif(nt*d,-a,a),nt,d)

#xtest = x

f = function(xinput){
   rad = sqrt(rowSums(xinput[,1:2]^2))
   prob = pnorm(sin(rad))
   alpha = 1
   prob = (xinput[,3]>1)*prob^alpha + (xinput[,3]<1)*(1-prob)^(alpha)
   prob = prob 
   return(prob)
}

k = 0.3
p = f(x)*k+ (1-k)*runif(nrow(x))
p = (p - min(p))/(max(p)-min(p))
y = (p > 0.5)*1
ytest = (f(xtest) > 0.5)*1

### XBART ###
get_XBART_params <- function(n,d,y){
    XBART_params = list(M = 15,
    L = 1,
    nsweeps = 150,
    Nmin = 100,
    alpha = 0.95,
    beta = 1.25,
    mtry = 10,
    burnin = 15)
    num_tress = XBART_params$M
    XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
    XBART_params$Ncutpoints = 100;XBART_params$tau = var(y)/(num_tress)
    XBART_params$a = 0.000001; XBART_params$b = 0.000001;
    return(XBART_params)
}
params = get_XBART_params(n,d,y)
dcat = 0
parl = F

t = proc.time()
out = XBART(as.matrix(y)-mean(y), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = params$Nmin, num_cutpoints = params$Ncutpoints,
alpha = params$alpha, beta = params$beta, tau = var(y)/params$M, s= 1,kap = 1,
mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
parallel = parl,random_seed = 10, no_split_penality = 0)
pred = predict(out,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.pred = apply(pred[,params$burnin:params$nsweeps],1,mean) + mean(y)


### CLT ###
get_XBART_params_clt <- function(n,d,y){
   XBART_params = list(M = 15,
   L = 1,
   nsweeps = 150,
   Nmin = 100,
   alpha = 0.95,
   beta = 1.25,
   mtry = 10,
   burnin = 15)
   num_tress = XBART_params$M
   XBART_params$max_depth = matrix(250, num_tress, XBART_params$nsweeps)
   XBART_params$Ncutpoints = 100;XBART_params$tau = var(y)/(num_tress)
   return(XBART_params)
}
params_clt = get_XBART_params(n,d,y)
t = proc.time()
out_clt = XBART.CLT(matrix(y)-mean(y), as.matrix(x), as.matrix(xtest), num_trees = params_clt$M, L = 1, num_sweeps = params_clt$nsweeps, max_depth = params_clt$max_depth, Nmin = params_clt$Nmin, num_cutpoints = params_clt$Ncutpoints,
alpha = params_clt$alpha, beta = params_clt$beta, tau = params_clt$tau, s= 1,kap = 1,
mtry = params_clt$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
parallel = parl,random_seed = 10)
pred.clt = predict(out_clt,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.clt.pred = apply(pred.clt[,params$burnin:params$nsweeps],1,mean)+mean(y)

### Probit ###
t = proc.time()
out_probit = XBART.Probit(as.matrix(y), as.matrix(x), as.matrix(xtest), num_trees = params$M, L = 1, num_sweeps = params$nsweeps, max_depth = params$max_depth, Nmin = 10, num_cutpoints = params$Ncutpoints,
alpha = params$alpha, beta = params$beta, tau = params$tau, s= 1,kap = 1,
mtry = params$mtry, p_categorical = dcat, draw_sigma = FALSE, m_update_sigma = TRUE,draw_mu= TRUE,
parallel = parl,random_seed = 10)
pred.probit = predict(out_probit,as.matrix(xtest))
t = proc.time() - t
print(t)
yhat.probit.pred = apply(pnorm(pred.probit[,params$burnin:params$nsweeps]),1,mean)

printTPR(yhat.pred,ytest)
printTPR(yhat.clt.pred ,ytest)
printTPR(yhat.probit.pred ,ytest)


print("Importance:")
print("XBART: ")
print(out$importance)
print("XBART CLT: ")
print(out_clt$importance)
print("XBART Probit: ")
print(out_probit$importance)






