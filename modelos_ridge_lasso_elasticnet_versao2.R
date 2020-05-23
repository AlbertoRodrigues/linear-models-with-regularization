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
X=model.matrix(y~x+I(x^2)+I(x^3))[,-1];X
dados=cbind.data.frame(X,y);dados

lambda = 10^seq(-4, 4, length = 200);lambda
#Regressão Ridge
ridge <- train(
  y ~., data = dados, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda),
  metric="MAE"
)
plot(ridge,sign.lambda = 1)
pos=which(ridge$results$lambda==ridge$bestTune$lambda)
ridge$results[pos,]
# Parâmetros
coef(ridge$finalModel, ridge$bestTune$lambda)


#Regressão Lasso
lasso <- train(
  y ~., data = dados, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda),
  metric="MAE"
)
plot(lasso,sign.lambda = 1)
pos=which(lasso$results$lambda==lasso$bestTune$lambda)
lasso$results[pos,]
# Parâmetros
coef(lasso$finalModel, lasso$bestTune$lambda)

#Elastic Net

elastic <- train(
  y ~., data =dados, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10,
  metric="MAE"
)
plot(elastic,sign.lambda = 1)
pos=which(elastic$results$alpha==elastic$bestTune$alpha & elastic$results$lambda==elastic$bestTune$lambda)
elastic$results[pos,]
# Parâmetros
coef(elastic$finalModel, elastic$bestTune$lambda)


models = list(ridge = ridge, lasso = lasso, elastic = elastic)
summary(resamples(models), metric = c("RMSE","MAE","Rsquared"))
