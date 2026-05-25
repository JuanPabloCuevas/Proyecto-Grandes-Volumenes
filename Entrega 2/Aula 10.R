# Librerías necesarias
library(MASS)
library(randomForest)
library(gbm)
library(xgboost)
library(Matrix)

# Fijar semilla para reproducibilidad
set.seed(2021)

# División de datos en entrenamiento y prueba
train <- sample(1:506, size = 374)
boston.test <- Boston[-train, "medv"]

# ----------------------------
# BAGGING (mtry = número total de variables)
# ----------------------------
bag.boston <- randomForest(medv ~ ., data = Boston, subset = train, 
                           mtry = 13, ntree = 5000, importance = TRUE)
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
mse.bag <- mean((yhat.bag - boston.test)^2)

# Gráfico y evaluación
plot(yhat.bag, boston.test, main = "Bagging: Predicción vs Real", xlab = "Predicción", ylab = "Real")
abline(0, 1, col = "red")
importance(bag.boston)
varImpPlot(bag.boston)

# ----------------------------
# RANDOM FOREST (mtry < número total de variables)
# ----------------------------
set.seed(2021)
rf.boston <- randomForest(medv ~ ., data = Boston, subset = train,
                          mtry = 11, ntree = 1000, importance = TRUE)
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mse.rf <- mean((yhat.rf - boston.test)^2)

importance(rf.boston)
varImpPlot(rf.boston)

# ----------------------------
# BOOSTING con GBM
# ----------------------------
set.seed(2021)
Boston.boost <- gbm(medv ~ ., data = Boston[train, ], distribution = "gaussian",
                    n.trees = 10000, shrinkage = 0.01, interaction.depth = 4, verbose = FALSE)
summary(Boston.boost)

n.trees <- seq(from = 100, to = 10000, by = 100)
predmatrix <- predict(Boston.boost, Boston[-train, ], n.trees = n.trees)
test.error <- with(Boston[-train, ], apply((predmatrix - medv)^2, 2, mean))
mse.boost <- min(test.error)

plot(n.trees, test.error, pch = 19, col = "blue", xlab = "Número de árboles", ylab = "Error cuadrático",
     main = "Boosting: Error vs Número de árboles")
abline(h = mse.rf, col = "red")
legend("topright", c("Random Forest"), col = "red", lty = 1, lwd = 2)

# ----------------------------
# XGBOOST
# ----------------------------
# Preparar matrices para XGBoost
train.data <- Boston[train, ]
test.data <- Boston[-train, ]
train_matrix <- sparse.model.matrix(medv ~ . -1, data = train.data)
test_matrix <- sparse.model.matrix(medv ~ . -1, data = test.data)

dtrain <- xgb.DMatrix(data = train_matrix, label = train.data$medv)
dtest <- xgb.DMatrix(data = test_matrix, label = test.data$medv)

# Entrenar el modelo XGBoost
set.seed(2021)
xgb.model <- xgboost(data = dtrain,
                     objective = "reg:squarederror",
                     nrounds = 500,
                     eta = 0.05,
                     max_depth = 4,
                     verbose = 0)

# Predicción y error
yhat.xgb <- predict(xgb.model, newdata = dtest)
mse.xgb <- mean((yhat.xgb - boston.test)^2)

# Importancia de variables
importance_matrix <- xgb.importance(model = xgb.model)
xgb.plot.importance(importance_matrix)

# ----------------------------
# COMPARACIÓN DE LOS MÉTODOS
# ----------------------------
cat("Error cuadrático medio (MSE) para cada método:\n")
cat("Bagging:        ", round(mse.bag, 4), "\n")
cat("Random Forest:  ", round(mse.rf, 4), "\n")
cat("Boosting (GBM): ", round(mse.boost, 4), "\n")
cat("XGBoost:        ", round(mse.xgb, 4), "\n")

