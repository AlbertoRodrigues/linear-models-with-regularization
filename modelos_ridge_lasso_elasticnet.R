require(glmnet)
require(caret)

set.seed(7)
x = rnorm(100, 2, 0.5)
e = rnorm(100)

b0=1.4
b1=2.5
b2=7.2
b3=1.8
y = b0+x*b1+(x^2)*b2+(x^3)*b3+e
modelo=lm(y~x+I(x^2)+I(x^3));modelo

X=model.matrix(modelo)[,-1];X

#Regressão Ridge
cv = cv.glmnet(X, y, alpha = 0,nfolds=10,type.measure = "mae")
cv$lambda.min
cv$cvm
cv$cvsd
cv$cvup
cv$cvlo

modelo = glmnet(X, y, alpha = 0, lambda = cv$lambda.min)

coef(modelo)

plot(cv,sign.lambda = 1)

#Regressão Lasso
cv = cv.glmnet(X, y, alpha = 1,nfolds=10,type.measure = "mae")
cv$lambda.min
cv$cvm
cv$cvsd
cv$cvup
cv$cvlo
modelo = glmnet(X, y, alpha = 1, lambda = cv$lambda.min)

coef(modelo)

#Elastic Net
dados=cbind.data.frame(X,y);dados
model <- train(
  y ~., data = dados, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 15
)
model$bestTune
model$bestTune$lambda
model$bestTune$alpha
l=which(model$results$alpha==model$bestTune$alpha & model$results$lambda==model$bestTune$lambda)
model$results[l,]$MAE

coef(model$finalModel, model$bestTune$lambda)