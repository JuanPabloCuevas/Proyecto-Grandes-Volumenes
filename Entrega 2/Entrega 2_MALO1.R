library(dplyr)
library(readr)
library(tidyr)

data <- read_csv("airline_2m.csv", show_col_types = FALSE)

# Selección inicial de variables relevantes para predecir ArrDel15.
# Se excluyen variables que generan fuga de información porque solo se conocen
# después del vuelo o están derivadas del resultado final.
variables_prevuelo <- c(
	"Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
	"Reporting_Airline",
	"OriginAirportID", "OriginCityMarketID", "OriginState", "OriginWac",
	"DestAirportID", "DestCityMarketID", "DestState", "DestWac",
	"CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
	"CRSElapsedTime", "Distance", "DistanceGroup"
)

variables_descartadas <- c(
	"DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
	"TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
	"ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrivalDelayGroups",
	"Cancelled", "CancellationCode", "Diverted",
	"ActualElapsedTime", "AirTime",
	"CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
	"FirstDepTime", "TotalAddGTime", "LongestAddGTime",
	"DivAirportLandings", "DivReachedDest", "DivActualElapsedTime", "DivArrDelay", "DivDistance",
	"Div1Airport", "Div1AirportID", "Div1AirportSeqID", "Div1WheelsOn", "Div1TotalGTime",
	"Div1LongestGTime", "Div1WheelsOff", "Div1TailNum",
	"Div2Airport", "Div2AirportID", "Div2AirportSeqID", "Div2WheelsOn", "Div2TotalGTime",
	"Div2LongestGTime", "Div2WheelsOff", "Div2TailNum",
	"Div3Airport", "Div3AirportID", "Div3AirportSeqID", "Div3WheelsOn", "Div3TotalGTime",
	"Div3LongestGTime", "Div3WheelsOff", "Div3TailNum",
	"Div4Airport", "Div4AirportID", "Div4AirportSeqID", "Div4WheelsOn", "Div4TotalGTime",
	"Div4LongestGTime", "Div4WheelsOff", "Div4TailNum",
	"Div5Airport", "Div5AirportID", "Div5AirportSeqID", "Div5WheelsOn", "Div5TotalGTime",
	"Div5LongestGTime", "Div5WheelsOff", "Div5TailNum"
)

data_modelo <- data %>%
	select(ArrDel15, any_of(variables_prevuelo)) %>%
	mutate(
		ArrDel15 = factor(ArrDel15, levels = c(0, 1), labels = c("No", "Si"))
	) %>%
	filter(!is.na(ArrDel15))

cat("Filas:", nrow(data_modelo), "Columnas:", ncol(data_modelo), "\n")
print(names(data_modelo))

# Preparación de datos para modelado
data_final <- data_modelo %>%
  mutate(
    Reporting_Airline = factor(Reporting_Airline),
    OriginState = factor(OriginState),
    DestState = factor(DestState),
    DayOfWeek = factor(DayOfWeek),
    Quarter = factor(Quarter),
    Month = factor(Month),
    DepTimeBlk = factor(DepTimeBlk),
    ArrTimeBlk = factor(ArrTimeBlk),
    DistanceGroup = factor(DistanceGroup)
  ) %>%
  drop_na()

cat("\n=== DATOS FINALES PARA MODELADO ===\n")
cat("Filas:", nrow(data_final), "\n")
cat("Columnas:", ncol(data_final), "\n")

# ============================================================================
# ANÁLISIS DESCRIPTIVO PREVIO: SELECCIÓN DE VARIABLES
# ============================================================================

cat("\n=== ANÁLISIS DE DATOS FALTANTES EN data_modelo (ANTES DE ELIMINAR) ===\n")
valores_faltantes_previo <- colSums(is.na(data_modelo))
valores_faltantes_previo <- valores_faltantes_previo[valores_faltantes_previo > 0]

if (length(valores_faltantes_previo) > 0) {
  cat("Columnas con NAs:\n")
  valores_faltantes_sorted <- sort(valores_faltantes_previo, decreasing = TRUE)
  for (i in seq_along(valores_faltantes_sorted)) {
    pct <- round(valores_faltantes_sorted[i] / nrow(data_modelo) * 100, 2)
    cat(sprintf("  %s: %d NAs (%.2f%%)\n", names(valores_faltantes_sorted)[i], 
                valores_faltantes_sorted[i], pct))
  }
} else {
  cat("✓ No hay datos faltantes\n")
}

# Correlación con la variable respuesta
cat("\n=== CORRELACIÓN DE VARIABLES NUMÉRICAS CON ArrDel15 ===\n")

# Crear una versión numérica de ArrDel15
data_correlacion <- data_modelo %>%
  mutate(ArrDel15_num = as.numeric(ArrDel15) - 1) %>%
  select(where(is.numeric)) %>%
  select(-ArrDel15_num)

if (ncol(data_correlacion) > 0) {
  # Convertir ArrDel15 a numérico para calcular correlación
  ArrDel15_num <- as.numeric(data_modelo$ArrDel15) - 1
  
  correlaciones <- sapply(data_correlacion, function(x) {
    cor(x, ArrDel15_num, use = "complete.obs")
  })
  
  correlaciones_sorted <- sort(abs(correlaciones), decreasing = TRUE)
  cat("Correlaciones ordenadas por magnitud:\n")
  for (i in seq_along(correlaciones_sorted)) {
    var_name <- names(correlaciones_sorted)[i]
    cor_value <- correlaciones[var_name]
    cat(sprintf("  %s: %.4f\n", var_name, cor_value))
  }
} else {
  cat("No hay variables numéricas para correlacionar (excluyendo ArrDel15)\n")
}

# Análisis de multicolinealidad entre predictoras
cat("\n=== ANÁLISIS DE MULTICOLINEALIDAD ===\n")
cat("(Justificación para usar Ridge en lugar de Lasso)\n\n")

# Calcular matriz de correlación entre variables numéricas
vars_numericas <- data_modelo %>%
  select(where(is.numeric)) %>%
  names()

if (length(vars_numericas) > 1) {
  cor_matrix <- cor(data_modelo[, vars_numericas], use = "complete.obs")
  
  # Detectar correlaciones altas (> 0.7 en valor absoluto)
  cor_altas <- which(abs(cor_matrix) > 0.7 & lower.tri(cor_matrix), arr.ind = TRUE)
  
  if (nrow(cor_altas) > 0) {
    cat("✓ Se detectó multicolinealidad (correlaciones > 0.7):\n")
    for (i in 1:nrow(cor_altas)) {
      row_idx <- cor_altas[i, 1]
      col_idx <- cor_altas[i, 2]
      var1 <- vars_numericas[row_idx]
      var2 <- vars_numericas[col_idx]
      cor_value <- cor_matrix[row_idx, col_idx]
      cat(sprintf("  %s <-> %s: %.4f\n", var1, var2, cor_value))
    }
    cat("\n➜ Ridge es la opción adecuada para manejar esta multicolinealidad.\n")
  } else {
    cat("⚠ No se detectaron correlaciones muy altas entre predictoras.\n")
    cat("  Sin embargo, Ridge sigue siendo útil para regularización general\n")
    cat("  y prevenir overfitting con múltiples variables.\n")
  }
} else {
  cat("No hay suficientes variables numéricas para analizar multicolinealidad\n")
}

# ============================================================================
# ETAPA 2: PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================

cat("\n=== ETAPA 2: PREPARACIÓN DE DATOS ===\n")

library(caret)

# 1. Verificación de datos faltantes
cat("\n=== VERIFICACIÓN DE DATOS FALTANTES ===\n")
valores_faltantes <- colSums(is.na(data_final))
if (sum(valores_faltantes) == 0) {
  cat("✓ No hay datos faltantes en data_final\n")
} else {
  cat("Columnas con datos faltantes:\n")
  print(valores_faltantes[valores_faltantes > 0])
}

# 2. Análisis de desbalanceo de clases
cat("\n=== DISTRIBUCIÓN DE LA VARIABLE RESPUESTA ===\n")
class_dist <- data_final %>%
  count(ArrDel15) %>%
  mutate(Porcentaje = round(n / sum(n) * 100, 2))
print(class_dist)

proporcion_clase_1 <- class_dist %>% filter(ArrDel15 == "Si") %>% pull(Porcentaje)
cat("Desbalanceo: ", proporcion_clase_1, "% de vuelos retrasados\n", sep = "")

# 3. Partición entrenamiento/prueba (80/20) usando createDataPartition
cat("\n=== PARTICIÓN ENTRENAMIENTO/PRUEBA (caret::createDataPartition) ===\n")
set.seed(2026)
trainIndex <- createDataPartition(
  data_final$ArrDel15, 
  p = 0.80, 
  list = FALSE, 
  times = 1
)

datos_train <- data_final[trainIndex, ]
datos_test <- data_final[-trainIndex, ]

cat("Datos de entrenamiento (antes de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Datos de prueba (antes de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# Tomar sample para reducir tamaño en memoria
cat("\n=== SAMPLING PARA MANEJO EFICIENTE DE MEMORIA ===\n")
set.seed(2026)

# Calcular tamaño de muestra fuera del pipe
n_train_sample <- min(100000, nrow(datos_train))
n_test_sample <- min(25000, nrow(datos_test))

datos_train <- datos_train %>%
  slice_sample(n = n_train_sample)

datos_test <- datos_test %>%
  slice_sample(n = n_test_sample)

cat("Datos de entrenamiento (después de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Datos de prueba (después de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# ============================================================================
# ETAPA 3: MODELO LOGÍSTICO
# ============================================================================

library(glmnet)
library(Matrix)

cat("\n=== MODELO LOGÍSTICO CON REGULARIZACIÓN (GLMNET) ===\n")

# Reducir cardinalidad: eliminar IDs de aeropuertos/ciudades que generan muchas columnas
# Mantener solo variables más interpretables y con menos niveles
variables_modelo <- c("ArrDel15", "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
                      "Reporting_Airline", "OriginState", "DestState",
                      "CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
                      "CRSElapsedTime", "Distance", "DistanceGroup")

datos_train_modelo <- datos_train %>% select(all_of(variables_modelo))
datos_test_modelo <- datos_test %>% select(all_of(variables_modelo))

# Combinar ambos conjuntos temporalmente para crear la matriz modelo completa
datos_temp <- bind_rows(datos_train_modelo, datos_test_modelo)

# Crear matriz modelo con datos combinados (esto garantiza columnas consistentes)
X_temp <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_temp)
y_temp <- as.numeric(datos_temp$ArrDel15) - 1

# Dividir la matriz combinada en entrenamiento y prueba
n_train <- nrow(datos_train_modelo)
n_test <- nrow(datos_test_modelo)

X_train_sparse <- X_temp[1:n_train, ]
y_train_binary <- y_temp[1:n_train]

X_test_sparse <- X_temp[(n_train + 1):(n_train + n_test), ]
y_test_binary <- y_temp[(n_train + 1):(n_train + n_test)]

cat("Matriz dispersa de entrenamiento: ", nrow(X_train_sparse), " x ", ncol(X_train_sparse), "\n", sep = "")
cat("Matriz dispersa de prueba: ", nrow(X_test_sparse), " x ", ncol(X_test_sparse), "\n", sep = "")

# Verificar que tienen el mismo número de columnas
if (ncol(X_train_sparse) != ncol(X_test_sparse)) {
  cat("⚠ ADVERTENCIA: Las matrices no tienen el mismo número de columnas\n")
} else {
  cat("✓ Las matrices tienen el mismo número de columnas\n")
}

# Ajustar modelo logístico con regularización L2 (Ridge)
set.seed(2026)

# Calcular pesos para balancear clases
# Peso = n_total / (2 * n_clase) para cada clase
n_total <- length(y_train_binary)
n_class_0 <- sum(y_train_binary == 0)
n_class_1 <- sum(y_train_binary == 1)

weight_0 <- n_total / (2 * n_class_0)
weight_1 <- n_total / (2 * n_class_1)

# Asignar pesos a cada observación
pesos <- ifelse(y_train_binary == 0, weight_0, weight_1)

cat("Pesos de clase:\n")
cat("Clase 0 (No retrasado):  ", round(weight_0, 3), "\n", sep = "")
cat("Clase 1 (Retrasado):     ", round(weight_1, 3), "\n", sep = "")

modelo_logistico <- glmnet(
  x = X_train_sparse,
  y = y_train_binary,
  weights = pesos,
  family = "binomial",
  alpha = 0,  # Ridge regression
  standardize = TRUE
)

cat("✓ Modelo entrenado con pesos de clase\n")
cat("Número de valores lambda probados:", length(modelo_logistico$lambda), "\n")

# Seleccionar lambda óptimo mediante validación cruzada (con pesos)
set.seed(2026)
cv_modelo <- cv.glmnet(
  x = X_train_sparse,
  y = y_train_binary,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  nfolds = 5
)

lambda_opt <- cv_modelo$lambda.min
cat("Lambda óptimo (mínimo error CV):", round(lambda_opt, 6), "\n")

# Predicciones en datos de entrenamiento
pred_train_prob <- predict(modelo_logistico, newx = X_train_sparse, 
                           s = lambda_opt, type = "response")[, 1]
pred_train_class <- ifelse(pred_train_prob > 0.5, "Si", "No")

# Predicciones en datos de prueba
pred_test_prob <- predict(modelo_logistico, newx = X_test_sparse, 
                          s = lambda_opt, type = "response")[, 1]
pred_test_class <- ifelse(pred_test_prob > 0.5, "Si", "No")

# Matrices de confusión
cat("\n=== DESEMPEÑO EN ENTRENAMIENTO ===\n")
cm_train <- table(Observado = datos_train_modelo$ArrDel15, Predicho = pred_train_class)
print(cm_train)

cat("\n=== DESEMPEÑO EN PRUEBA ===\n")
cm_test <- table(Observado = datos_test_modelo$ArrDel15, Predicho = pred_test_class)
print(cm_test)

# Métricas de desempeño
source_code <- "
accuracy <- function(cm) {
  (cm[1,1] + cm[2,2]) / sum(cm)
}
precision <- function(cm) {
  cm[2,2] / (cm[2,2] + cm[1,2])
}
recall <- function(cm) {
  cm[2,2] / (cm[2,2] + cm[2,1])
}
f1_score <- function(cm) {
  prec <- precision(cm)
  rec <- recall(cm)
  2 * (prec * rec) / (prec + rec)
}
mse <- function(y_obs, y_pred) {
  mean((y_obs - y_pred)^2)
}
rmse <- function(y_obs, y_pred) {
  sqrt(mean((y_obs - y_pred)^2))
}
"
eval(parse(text = source_code))

cat("\n=== MÉTRICAS ENTRENAMIENTO ===\n")
cat("Accuracy:  ", round(accuracy(cm_train), 4), "\n", sep = "")
cat("Precision: ", round(precision(cm_train), 4), "\n", sep = "")
cat("Recall:    ", round(recall(cm_train), 4), "\n", sep = "")
cat("F1-Score:  ", round(f1_score(cm_train), 4), "\n", sep = "")
cat("MSE:       ", round(mse(y_train_binary, pred_train_prob), 4), "\n", sep = "")
cat("RMSE:      ", round(rmse(y_train_binary, pred_train_prob), 4), "\n", sep = "")

cat("\n=== MÉTRICAS PRUEBA ===\n")
cat("Accuracy:  ", round(accuracy(cm_test), 4), "\n", sep = "")
cat("Precision: ", round(precision(cm_test), 4), "\n", sep = "")
cat("Recall:    ", round(recall(cm_test), 4), "\n", sep = "")
cat("F1-Score:  ", round(f1_score(cm_test), 4), "\n", sep = "")
cat("MSE:       ", round(mse(y_test_binary, pred_test_prob), 4), "\n", sep = "")
cat("RMSE:      ", round(rmse(y_test_binary, pred_test_prob), 4), "\n", sep = "")

