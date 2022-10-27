library(XBART)
print("Read Data")
train_fe = read.csv("train_fe.csv")
valid_fe = read.csv("valid_fe.csv")
X_train = train_fe[,!(names(train_fe) %in% c("Survived"))]
X_test = valid_fe[,!(names(valid_fe) %in% c("Survived"))]
y_train = train_fe$Survived
y_test = valid_fe$Survived


num_trees = 100
num_sweeps = 100
burnin = 15
fit = XBART.multinomial(y=matrix(train_fe$Survived), num_class=2, 
						X=X_train, Xtest=X_test, num_trees=num_trees, 
						num_sweeps=num_sweeps, max_depth=250, 
						Nmin=1, num_cutpoints=250, alpha=0.95 ,
						beta=1.25, tau=1/num_trees,
            			no_split_penality = 0.5, burnin = burnin) 

a = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), median)
pred = apply(a,1,which.max)-1
logLoss = function(pred, actual){
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

logLoss(a[,1], y_test)

