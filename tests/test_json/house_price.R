library("XBART")
test = read.csv("test.csv")

model = load.XBART("model.xbart")

test_data = test[ , !(names(test) %in% c("target"))]
pred2 = predict(model, as.matrix(test_data))
pred2 = rowMeans(pred2[, 16:dim(pred2)[2]])

print(paste("rmse of fit xbart loaded into R: ", sqrt(mean((pred2 - test["target"]) ^ 2))))