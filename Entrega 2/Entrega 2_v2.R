# ============================================================================
# ANÁLISIS PREDICTIVO: RETRASOS AÉREOS (ArrDel15)
# Versión 2: Variables Interpretables desde el Inicio
# 
# Etapa 1: Carga y preparación de datos
# Etapa 2: Creación de variables interpretables
# Etapa 3: Análisis descriptivo previo a modelado
# Etapa 4: Partición y preparación para modelado
# Etapa 5: Modelo logístico con Ridge
# Etapa 6: Modelo árbol de clasificación
# Etapa 7: Random Forest
# Etapa 8: Gradient Boosting (GBM)
# Etapa 9: XGBoost
# Etapa 10: Comparación de todos los modelos
# ============================================================================


# ============================================================================
# LIBRERÍAS
# ============================================================================
library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(pROC)
library(glmnet)
library(Matrix)
library(rpart)
library(rpart.plot)
library(data.table)
library(ranger)
library(gbm)
library(xgboost)
library(parallel)

# ============================================================================
# ETAPA 1: CARGA Y PREPARACIÓN INICIAL DE DATOS
# ============================================================================

cat("\n=== ETAPA 1: CARGA Y PREPARACIÓN DE DATOS ===\n")

# Lectura de datos
data <- read_csv("airline_2m.csv", show_col_types = FALSE)

# Transformar a data table 
setDT(data) 

# Definición de variables prevuelo (sin fuga de información)
variables_prevuelo <- c(
  "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
  "Reporting_Airline",
  "OriginAirportID", "OriginCityMarketID", "OriginState", "OriginWac",
  "DestAirportID", "DestCityMarketID", "DestState", "DestWac",
  "CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
  "CRSElapsedTime", "Distance", "DistanceGroup"
)


# Selección inicial de variables y conversión de variable respuesta
data_modelo <- data %>%
  select(ArrDel15, any_of(variables_prevuelo)) %>%
  mutate(ArrDel15 = factor(ArrDel15, levels = c(0, 1), labels = c("No", "Si"))) %>%
  filter(!is.na(ArrDel15))

cat("Datos iniciales: ", nrow(data_modelo), " filas, ", ncol(data_modelo), " columnas\n", sep = "")

# Conversión de variables categóricas y eliminación de NAs
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

# Análisis de datos faltantes ANTES de eliminar
cat("\n=== ANÁLISIS DE DATOS FALTANTES (previo a limpieza) ===\n")
valores_faltantes_previo <- colSums(is.na(data_final))
valores_faltantes_previo <- valores_faltantes_previo[valores_faltantes_previo > 0]

if (length(valores_faltantes_previo) > 0) {
  cat("Columnas con NAs:\n")
  valores_faltantes_sorted <- sort(valores_faltantes_previo, decreasing = TRUE)
  for (i in seq_along(valores_faltantes_sorted)) {
    pct <- round(valores_faltantes_sorted[i] / nrow(data_final) * 100, 2)
    cat(sprintf("  %s: %d NAs (%.2f%%)\n", names(valores_faltantes_sorted)[i], 
                valores_faltantes_sorted[i], pct))
  }
} else {
  cat("✓ No hay datos faltantes\n")
}

cat("\nDatos después de limpieza: ", nrow(data_final), " filas\n", sep = "")

# ============================================================================
# ETAPA 2: CREACIÓN DE VARIABLES INTERPRETABLES
# ============================================================================

cat("\n=== ETAPA 2: CREACIÓN DE VARIABLES INTERPRETABLES ===\n")

# Función para categorizar riesgo basado en tasa de retrasos por aeropuerto
categorizar_riesgo <- function(data, id_col, target_col) {
  tasas <- data %>%
    group_by(!!sym(id_col)) %>%
    summarise(
      tasa_retraso = mean(as.numeric(!!sym(target_col)) - 1, na.rm = TRUE),
      .groups = "drop"
    )
  
  q33 <- quantile(tasas$tasa_retraso, 0.33, na.rm = TRUE)
  q67 <- quantile(tasas$tasa_retraso, 0.67, na.rm = TRUE)
  
  tasas <- tasas %>%
    mutate(
      riesgo = case_when(
        tasa_retraso <= q33 ~ "Bajo",
        tasa_retraso <= q67 ~ "Medio",
        TRUE ~ "Alto"
      )
    )
  
  return(tasas)
}

# Función para categorizar hora del día (VECTORIZADA)
categorizar_hora <- function(hora) {
  # Extraer horas de forma vectorizada
  hora_num <- floor(as.numeric(hora) / 100)
  
  case_when(
    is.na(hora_num) ~ NA_character_,
    hora_num >= 6 & hora_num < 12 ~ "Mañana",
    hora_num >= 12 & hora_num < 18 ~ "Tarde",
    hora_num >= 18 & hora_num < 24 ~ "Noche",
    TRUE ~ "Madrugada"
  )
}

# Función para categorizar estación
categorizar_estacion <- function(mes) {
  case_when(
    mes %in% c(12, 1, 2) ~ "Invierno",
    mes %in% c(3, 4, 5) ~ "Primavera",
    mes %in% c(6, 7, 8) ~ "Verano",
    mes %in% c(9, 10, 11) ~ "Otoño",
    TRUE ~ NA_character_
  )
}

# Función para categorizar tiempo de vuelo
categorizar_tiempo_grupo <- function(minutos) {
  case_when(
    is.na(minutos) ~ NA_character_,
    minutos < 120 ~ "MuyCorto",
    minutos < 180 ~ "Corto",
    minutos < 300 ~ "Medio",
    minutos < 420 ~ "Largo",
    TRUE ~ "MuyLargo"
  )
}


# Categorizar aeropuertos, aerolínea, distancia y tiempo
# sacaría riesgo origen porque no nos interesa si los vuelos llegan tarde
# en el aeropuerto de origen
riesgo_origen <- categorizar_riesgo(data_final, "OriginAirportID", "ArrDel15")
riesgo_destino <- categorizar_riesgo(data_final, "DestAirportID", "ArrDel15")
riesgo_aerolinea <- categorizar_riesgo(data_final, "Reporting_Airline", "ArrDel15")
riesgo_distancia <- categorizar_riesgo(data_final, "DistanceGroup", "ArrDel15")

# Crear categorías de tiempo en tabla temporal para calcular riesgo
tiempo_temp <- data_final %>%
  mutate(TiempoGrupo = factor(categorizar_tiempo_grupo(CRSElapsedTime)))
riesgo_tiempo <- categorizar_riesgo(tiempo_temp, "TiempoGrupo", "ArrDel15")

cat("  ✓ Origen: ", paste(table(riesgo_origen$riesgo), collapse = " | "), "\n", sep = "")
cat("  ✓ Destino: ", paste(table(riesgo_destino$riesgo), collapse = " | "), "\n", sep = "")
cat("  ✓ Aerolínea: ", paste(table(riesgo_aerolinea$riesgo), collapse = " | "), "\n", sep = "")
cat("  ✓ Distancia: ", paste(table(riesgo_distancia$riesgo), collapse = " | "), "\n", sep = "")
cat("  ✓ Tiempo: ", paste(table(riesgo_tiempo$riesgo), collapse = " | "), "\n", sep = "")

# Crear nueva tabla con variables interpretables
data_interpretable <- data_final %>%
  mutate(TiempoGrupo = factor(categorizar_tiempo_grupo(CRSElapsedTime))) %>%
  left_join(riesgo_origen %>% select(OriginAirportID, riesgo),
            by = "OriginAirportID", suffix = c("", "_origen")) %>%
  rename(OriginRiesgo = riesgo) %>%
  left_join(riesgo_destino %>% select(DestAirportID, riesgo),
            by = "DestAirportID", suffix = c("", "_destino")) %>%
  rename(DestRiesgo = riesgo) %>%
  left_join(riesgo_aerolinea %>% select(Reporting_Airline, riesgo),
            by = "Reporting_Airline", suffix = c("", "_aerolinea")) %>%
  rename(AerolineaRiesgo = riesgo) %>%
  left_join(riesgo_distancia %>% select(DistanceGroup, riesgo),
            by = "DistanceGroup", suffix = c("", "_distancia")) %>%
  rename(DistanciaRiesgo = riesgo) %>%
  left_join(riesgo_tiempo %>% select(TiempoGrupo, riesgo),
            by = "TiempoGrupo", suffix = c("", "_tiempo")) %>%
  rename(TiempoRiesgo = riesgo) %>%
  mutate(
    OriginRiesgo = factor(OriginRiesgo, levels = c("Bajo", "Medio", "Alto")),
    DestRiesgo = factor(DestRiesgo, levels = c("Bajo", "Medio", "Alto")),
    AerolineaRiesgo = factor(AerolineaRiesgo, levels = c("Bajo", "Medio", "Alto")),
    DistanciaRiesgo = factor(DistanciaRiesgo, levels = c("Bajo", "Medio", "Alto")),
    TiempoRiesgo = factor(TiempoRiesgo, levels = c("Bajo", "Medio", "Alto")),
    HoraSalida = factor(categorizar_hora(CRSDepTime), levels = c("Madrugada", "Mañana", "Tarde", "Noche")),
    HoraLlegada = factor(categorizar_hora(CRSArrTime), levels = c("Madrugada", "Mañana", "Tarde", "Noche")),
    Estacion = factor(categorizar_estacion(Month), levels = c("Primavera", "Verano", "Otoño", "Invierno"))
  ) %>%
  select(-OriginAirportID, -DestAirportID, -CRSDepTime, -CRSArrTime, -Month, 
         -OriginCityMarketID, -DestCityMarketID, -OriginWac, -DestWac, -DepTimeBlk, 
         -ArrTimeBlk, -OriginState, -DestState, -Reporting_Airline, -Distance, 
         -DistanceGroup, -CRSElapsedTime, -TiempoGrupo, -Quarter, -DayofMonth)

cat("✓ Variables interpretables creadas en data_interpretable\n")

# ============================================================================
# ETAPA 3: ANÁLISIS DESCRIPTIVO PREVIO A MODELADO
# ============================================================================

cat("\n=== ETAPA 3: ANÁLISIS DESCRIPTIVO ===\n")

# Distribución de variable respuesta
cat("\n--- Distribución de la variable respuesta ---\n")
class_dist <- data_interpretable %>%
  count(ArrDel15) %>%
  mutate(Porcentaje = round(n / sum(n) * 100, 2))
print(class_dist)

# Correlación con variable respuesta
cat("\n--- Análisis de variables numéricas ---\n")
vars_numericas <- data_interpretable %>% select(where(is.numeric)) %>% names()

if (length(vars_numericas) > 0) {
  cat("Correlaciones con ArrDel15:\n")
  ArrDel15_num <- as.numeric(data_interpretable$ArrDel15) - 1
  
  correlaciones <- sapply(vars_numericas, function(var) {
    cor(data_interpretable[[var]], ArrDel15_num, use = "complete.obs")
  })
  
  correlaciones_sorted <- sort(abs(correlaciones), decreasing = TRUE)
  for (i in seq_along(correlaciones_sorted)) {
    var_name <- names(correlaciones_sorted)[i]
    cor_value <- correlaciones[var_name]
    cat(sprintf("  %s: %.4f\n", var_name, cor_value))
  }
}

# ============================================================================
# ETAPA 4: PARTICIÓN Y PREPARACIÓN PARA MODELADO
# ============================================================================

cat("\n=== ETAPA 4: PREPARACIÓN DE DATOS PARA MODELADO ===\n")
set.seed(2026)
#sampling
n_train <- 100000
n_test <- 25000
n_muestra_total <- n_train + n_test

#sampling toda la muestra
data_muestra <- data_final %>% slice_sample(n = n_muestra_total)

# Partición estratificada
trainIndex <- createDataPartition(
  data_muestra$ArrDel15,
  p = 0.80,
  list = FALSE,
  times = 1
)

datos_train <- data_muestra[trainIndex, ]
datos_test <- data_muestra[-trainIndex, ]

cat("Entrenamiento (después de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Prueba (después de sampling):        ", nrow(datos_test), " filas\n", sep = "")

#verificar que las proporciones se mantuvieron
cat("\n--- Proporción de retrasos en entrenamiento ---\n")
print(prop.table(table(datos_train$ArrDel15)))

cat("\n--- Proporción de retrasos en prueba ---\n")
print(prop.table(table(datos_test$ArrDel15)))

# ============================================================================
#  MODELO LOGÍSTICO Y CON REGULARIZACIÓN RIDGE
# ============================================================================


cat("\n=== MODELO LOGÍSTICO===\n")

# Combinar para matriz modelo consistente
datos_temp <- bind_rows(datos_train, datos_test)
X_temp <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_temp)
y_temp <- as.numeric(datos_temp$ArrDel15) - 1

n_train <- nrow(datos_train)
n_test <- nrow(datos_test)

X_train <- X_temp[1:n_train, ]
y_train <- y_temp[1:n_train]
X_test <- X_temp[(n_train + 1):(n_train + n_test), ]
y_test <- y_temp[(n_train + 1):(n_train + n_test)]

cat("Matriz dispersa - Entrenamiento: ", nrow(X_train), " x ", ncol(X_train), "\n", sep = "")

# Pesos de clase
n_total <- length(y_train)
n_class_0 <- sum(y_train == 0)
n_class_1 <- sum(y_train == 1)

weight_0 <- n_total / (2 * n_class_0)
weight_1 <- n_total / (2 * n_class_1)
pesos <- ifelse(y_train == 0, weight_0, weight_1)

cat("Pesos de clase:\n")
cat("  Clase 0 (No): ", round(weight_0, 3), "\n", sep = "")
cat("  Clase 1 (Si): ", round(weight_1, 3), "\n", sep = "")


#### MODELO LOGÍSTICO ####

# Entrenar logística
set.seed(2026)
modelo_logistico <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  standardize = TRUE,
  lambda = 0 
)

# Usamos type = "response" para obtener probabilidades entre 0 y 1
prediccion_logistica <- predict(modelo_logistico, 
                                newx = X_test, type = "response")

# Usamos 0.5 como umbral estándar inicial
predicciones_clase <- ifelse(prediccion_logistica > 0.5, 1, 0)

pred_factor <- factor(predicciones_clase, levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

# Matriz de confusión
matriz_confusion <- confusionMatrix(data = pred_factor, 
                                    reference = y_test_factor, 
                                    positive = "1")

cat("\n=== EVALUACIÓN - REGRESIÓN LOGÍSTICA ===\n")
print(matriz_confusion)

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_logistica <- matriz_confusion$table[2, 2]
fn_logistica <- matriz_confusion$table[1, 2]
fp_logistica <- matriz_confusion$table[2, 1]
tn_logistica <- matriz_confusion$table[1, 1]
sensibilidad_logistica <- tp_logistica / (tp_logistica + fn_logistica)
especificidad_logistica <- tn_logistica / (tn_logistica + fp_logistica)
precision_logistica <- tp_logistica / (tp_logistica + fp_logistica)
f1_logistica <- 2 * (precision_logistica * sensibilidad_logistica) / (precision_logistica + sensibilidad_logistica)

# AUC
curva_roc <- roc(response = y_test, predictor = as.numeric(prediccion_logistica))
auc_logistica <- auc(curva_roc)[1]

cat("\nSensibilidad (Recall): ", round(sensibilidad_logistica * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_logistica * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_logistica * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_logistica, 4), "\n", sep = "")
cat("AUC: ", round(auc_logistica, 4), "\n", sep = "")

# Entrenar Ridge
modelo_ridge <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  standardize = TRUE
)

# Validación cruzada para obtener lambda óptimo
cv_modelo <- cv.glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  nfolds = 5
)

lambda_optimo <- cv_modelo$lambda.min

# Predicciones con Ridge usando lambda óptimo
prediccion_ridge <- predict(modelo_ridge, 
                            newx = X_test, type = "response", s = lambda_optimo)

# Convertir a clases
predicciones_clase_ridge <- ifelse(prediccion_ridge > 0.5, 1, 0)
pred_factor_ridge <- factor(predicciones_clase_ridge, levels = c(0, 1))

# Matriz de confusión
matriz_confusion_ridge <- confusionMatrix(data = pred_factor_ridge, 
                                          reference = y_test_factor, 
                                          positive = "1")

cat("\n=== EVALUACIÓN - RIDGE ===\n")
print(matriz_confusion_ridge)

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_ridge <- matriz_confusion_ridge$table[2, 2]
fn_ridge <- matriz_confusion_ridge$table[1, 2]
fp_ridge <- matriz_confusion_ridge$table[2, 1]
tn_ridge <- matriz_confusion_ridge$table[1, 1]
sensibilidad_ridge <- tp_ridge / (tp_ridge + fn_ridge)
especificidad_ridge <- tn_ridge / (tn_ridge + fp_ridge)
precision_ridge <- tp_ridge / (tp_ridge + fp_ridge)
f1_ridge <- 2 * (precision_ridge * sensibilidad_ridge) / (precision_ridge + sensibilidad_ridge)

# AUC
curva_roc_ridge <- roc(response = y_test, predictor = as.numeric(prediccion_ridge))
auc_ridge <- auc(curva_roc_ridge)[1]

cat("\nSensibilidad (Recall): ", round(sensibilidad_ridge * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_ridge * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_ridge * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_ridge, 4), "\n", sep = "")
cat("AUC: ", round(auc_ridge, 4), "\n", sep = "")

cat("\n✓ Modelo Ridge entrenado\n")
cat("  Lambda óptimo: ", round(lambda_optimo, 6), "\n", sep = "")

# ============================================================================
# ETAPA 6: MODELO DE ÁRBOL DE CLASIFICACIÓN (RPART)
# ============================================================================

cat("\n=== ETAPA 6: ÁRBOL DE CLASIFICACIÓN (RPART) ===\n")

# Preparar datos para rpart
datos_train_rpart <- datos_train
datos_test_rpart <- datos_test

# Sincronizar niveles de factor
for (col in names(datos_test_rpart)) {
  if (is.factor(datos_test_rpart[[col]])) {
    levels(datos_test_rpart[[col]]) <- levels(datos_train_rpart[[col]])
  }
}

cat("\n--- Distribución de clases ---\n")
print(table(datos_train_rpart$ArrDel15))

# Pesos de clase para rpart
n_train_rpart <- nrow(datos_train_rpart)
n_no <- sum(datos_train_rpart$ArrDel15 == "No")
n_si <- sum(datos_train_rpart$ArrDel15 == "Si")

weight_no <- n_train_rpart / (2 * n_no)
weight_si <- n_train_rpart / (2 * n_si)
pesos_rpart <- ifelse(datos_train_rpart$ArrDel15 == "No", weight_no, weight_si)

cat("\nPesos de clase:\n")
cat("  Clase No: ", round(weight_no, 3), "\n", sep = "")
cat("  Clase Si: ", round(weight_si, 3), "\n", sep = "")

# Entrenar árbol
cat("\nEntrenando árbol...\n")
arbol_fit <- rpart(
  ArrDel15 ~ .,
  method = "class",
  data = datos_train_rpart,
  weights = pesos_rpart,
  control = rpart.control(cp = 0.001, minsplit = 10, minbucket = 5, xval = 5)
)

cat("✓ Árbol entrenado\n")

# Seleccionar CP óptimo
cp_optimo <- arbol_fit$cptable[which.min(arbol_fit$cptable[, "xerror"]), "CP"]
xerror_min <- min(arbol_fit$cptable[, "xerror"])

cat("CP óptimo: ", round(cp_optimo, 6), "\n", sep = "")
cat("Xerror mínimo: ", round(xerror_min, 6), "\n", sep = "")

# Podar árbol
arbol_pruned <- rpart::prune(arbol_fit, cp = cp_optimo)
cat("✓ Árbol podado\n")

# Visualizar
cat("\nGenerando visualización del árbol...\n")
rpart.plot(arbol_pruned, main = "Árbol de Clasificación - Retrasos Aéreos", cex = 0.6)

# Predicciones y evaluación
cat("\n--- Evaluación en datos de prueba ---\n")
predicciones_arbol <- predict(arbol_pruned, newdata = datos_test_rpart, type = "class")
predicciones_arbol_prob <- predict(arbol_pruned, newdata = datos_test_rpart, type = "prob")[, 2]

confusion_matrix <- table(Predicho = predicciones_arbol, Real = datos_test_rpart$ArrDel15)
print(confusion_matrix)

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_arbol <- confusion_matrix["Si", "Si"]
fn_arbol <- confusion_matrix["No", "Si"]
fp_arbol <- confusion_matrix["Si", "No"]
tn_arbol <- confusion_matrix["No", "No"]
sensibilidad_arbol <- tp_arbol / (tp_arbol + fn_arbol)
especificidad_arbol <- tn_arbol / (tn_arbol + fp_arbol)
precision_arbol <- tp_arbol / (tp_arbol + fp_arbol)
f1_arbol <- 2 * (precision_arbol * sensibilidad_arbol) / (precision_arbol + sensibilidad_arbol)

cat("\nSensibilidad (Recall) del árbol: ", round(sensibilidad_arbol * 100, 2), "%\n", sep = "")
cat("Especificidad del árbol: ", round(especificidad_arbol * 100, 2), "%\n", sep = "")
cat("Precisión del árbol: ", round(precision_arbol * 100, 2), "%\n", sep = "")
cat("F1-Score del árbol: ", round(f1_arbol, 4), "\n", sep = "")

# Almacenar AUC del árbol
roc_arbol <- roc(datos_test_rpart$ArrDel15, predicciones_arbol_prob)
auc_arbol <- roc_arbol$auc[1]
cat("AUC del árbol: ", round(auc_arbol, 4), "\n", sep = "")

# ============================================================================
# ETAPA 7: RANDOM FOREST (RANGER - Optimizado para datos grandes)
# ============================================================================

cat("\n=== ETAPA 7: RANDOM FOREST (RANGER) ===")
cat("\nEntrenando Random Forest con ranger...\n")

set.seed(2026)
rf_model <- ranger(
  ArrDel15 ~ .,
  data = datos_train_rpart,
  num.trees = 100,
  mtry = floor(sqrt(ncol(datos_train_rpart) - 1)),
  min.node.size = 20,
  sample.fraction = 0.7,
  importance = "impurity",
  probability = TRUE,
  case.weights = pesos_rpart,
  num.threads = parallel::detectCores()
)

cat("✓ Random Forest entrenado\n")

# Predicciones
rf_pred_obj <- predict(rf_model, data = datos_test_rpart)
rf_pred_prob <- rf_pred_obj$predictions[, 2]
rf_pred <- factor(ifelse(rf_pred_prob > 0.5, "Si", "No"), levels = c("No", "Si"))

# Evaluación
rf_cm <- table(Predicho = rf_pred, Real = datos_test_rpart$ArrDel15)
roc_rf <- roc(datos_test_rpart$ArrDel15, rf_pred_prob)
auc_rf <- roc_rf$auc[1]

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_rf <- rf_cm["Si", "Si"]
fn_rf <- rf_cm["No", "Si"]
fp_rf <- rf_cm["Si", "No"]
tn_rf <- rf_cm["No", "No"]
sensibilidad_rf <- tp_rf / (tp_rf + fn_rf)
especificidad_rf <- tn_rf / (tn_rf + fp_rf)
precision_rf <- tp_rf / (tp_rf + fp_rf)
f1_rf <- 2 * (precision_rf * sensibilidad_rf) / (precision_rf + sensibilidad_rf)

cat("\n--- Evaluación Random Forest ---\n")
print(rf_cm)
cat("\nSensibilidad (Recall): ", round(sensibilidad_rf * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_rf * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_rf * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_rf, 4), "\n", sep = "")
cat("AUC: ", round(auc_rf, 4), "\n", sep = "")

# Importancia de variables
cat("\n--- Top 10 variables más importantes ---\n")
rf_importance_sorted <- sort(rf_model$variable.importance, decreasing = TRUE)[1:10]
print(rf_importance_sorted)

# ============================================================================
# ETAPA 8: GRADIENT BOOSTING (GBM)
# ============================================================================

cat("\n\n=== ETAPA 8: GRADIENT BOOSTING (GBM) ===")
cat("\nEntrenando GBM...\n")

set.seed(2026)

# Preparar datos para GBM
datos_train_gbm <- datos_train_rpart %>%
  mutate(ArrDel15_num = as.numeric(ArrDel15) - 1)

datos_test_gbm <- datos_test_rpart %>%
  mutate(ArrDel15_num = as.numeric(ArrDel15) - 1)

gbm_model <- gbm(
  ArrDel15_num ~ . - ArrDel15,
  data = datos_train_gbm,
  distribution = "bernoulli",
  n.trees = 500,
  shrinkage = 0.01,
  interaction.depth = 4,
  weights = pesos_rpart,
  verbose = FALSE
)

cat("✓ GBM entrenado\n")

# Predicciones
gbm_pred_prob <- predict(gbm_model, newdata = datos_test_gbm, n.trees = 500, type = "response")
gbm_pred <- factor(ifelse(gbm_pred_prob > 0.5, "Si", "No"), levels = c("No", "Si"))

# Evaluación
gbm_cm <- table(Predicho = gbm_pred, Real = datos_test_gbm$ArrDel15)
roc_gbm <- roc(datos_test_gbm$ArrDel15, gbm_pred_prob)
auc_gbm <- roc_gbm$auc[1]

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_gbm <- gbm_cm["Si", "Si"]
fn_gbm <- gbm_cm["No", "Si"]
fp_gbm <- gbm_cm["Si", "No"]
tn_gbm <- gbm_cm["No", "No"]
sensibilidad_gbm <- tp_gbm / (tp_gbm + fn_gbm)
especificidad_gbm <- tn_gbm / (tn_gbm + fp_gbm)
precision_gbm <- tp_gbm / (tp_gbm + fp_gbm)
f1_gbm <- 2 * (precision_gbm * sensibilidad_gbm) / (precision_gbm + sensibilidad_gbm)

cat("\n--- Evaluación GBM ---\n")
print(gbm_cm)
cat("\nSensibilidad (Recall): ", round(sensibilidad_gbm * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_gbm * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_gbm * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_gbm, 4), "\n", sep = "")
cat("AUC: ", round(auc_gbm, 4), "\n", sep = "")

# Importancia de variables
cat("\n--- Top 10 variables más importantes ---\n")
gbm_importance <- summary(gbm_model, n.trees = 500, plotit = FALSE)
print(head(gbm_importance, 10))

# ============================================================================
# ETAPA 9: XGBOOST
# ============================================================================

cat("\n\n=== ETAPA 9: XGBOOST ===")
cat("\nPreparando matrices para XGBoost...\n")

# Preparar matrices dispersas
X_train_xgb <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_train_rpart)
X_test_xgb <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_test_rpart)

y_train_xgb <- as.numeric(datos_train_rpart$ArrDel15) - 1
y_test_xgb <- as.numeric(datos_test_rpart$ArrDel15) - 1

dtrain_xgb <- xgb.DMatrix(data = X_train_xgb, label = y_train_xgb, weight = pesos_rpart)
dtest_xgb <- xgb.DMatrix(data = X_test_xgb, label = y_test_xgb)

cat("✓ Matrices preparadas\n")
cat("  Entrenamiento: ", nrow(X_train_xgb), " x ", ncol(X_train_xgb), "\n", sep = "")
cat("  Prueba: ", nrow(X_test_xgb), " x ", ncol(X_test_xgb), "\n", sep = "")

# Pesos de clase para XGBoost
scale_pos_weight <- sum(y_train_xgb == 0) / sum(y_train_xgb == 1)

cat("\nEntrenando XGBoost...\n")

set.seed(2026)
modelo_xgb <- xgboost(
  x = X_train_xgb,
  y = y_train_xgb,
  weight = pesos_rpart,
  nrounds = 100,
  objective = "reg:squarederror",
  max_depth = 6,
  learning_rate = 0.05,
  subsample = 0.7,
  colsample_bytree = 0.7
)

cat("✓ XGBoost entrenado\n")

# Predicciones
xgb_pred_prob <- predict(modelo_xgb, newdata = X_test_xgb)
xgb_pred <- factor(ifelse(xgb_pred_prob > 0.5, "Si", "No"), levels = c("No", "Si"))

# Evaluación
xgb_cm <- table(Predicho = xgb_pred, Real = datos_test_rpart$ArrDel15)
roc_xgb <- roc(datos_test_rpart$ArrDel15, xgb_pred_prob)
auc_xgb <- roc_xgb$auc[1]

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_xgb <- xgb_cm["Si", "Si"]
fn_xgb <- xgb_cm["No", "Si"]
fp_xgb <- xgb_cm["Si", "No"]
tn_xgb <- xgb_cm["No", "No"]
sensibilidad_xgb <- tp_xgb / (tp_xgb + fn_xgb)
especificidad_xgb <- tn_xgb / (tn_xgb + fp_xgb)
precision_xgb <- tp_xgb / (tp_xgb + fp_xgb)
f1_xgb <- 2 * (precision_xgb * sensibilidad_xgb) / (precision_xgb + sensibilidad_xgb)

cat("\n--- Evaluación XGBoost ---\n")
print(xgb_cm)
cat("\nSensibilidad (Recall): ", round(sensibilidad_xgb * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_xgb * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_xgb * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_xgb, 4), "\n", sep = "")
cat("AUC: ", round(auc_xgb, 4), "\n", sep = "")

# Importancia de variables
cat("\n--- Top 10 variables más importantes ---\n")
xgb_importance <- xgb.importance(model = modelo_xgb)
print(head(xgb_importance, 10))

# ============================================================================
# ETAPA 10: COMPARACIÓN DE TODOS LOS MODELOS
# ============================================================================

cat("\n\n=== ETAPA 10: COMPARACIÓN DE MODELOS ===")
cat("\n--- Resumen de desempeño de todos los modelos ---\n")

# Crear tabla comparativa
comparacion <- data.frame(
  Modelo = c("Logística", "Ridge", "Árbol (RPART)", "Random Forest", "GBM", "XGBoost"),
  Sensibilidad = c(
    round(sensibilidad_logistica * 100, 2),
    round(sensibilidad_ridge * 100, 2),
    round(sensibilidad_arbol * 100, 2),
    round(sensibilidad_rf * 100, 2),
    round(sensibilidad_gbm * 100, 2),
    round(sensibilidad_xgb * 100, 2)
  ),
  Especificidad = c(
    round(especificidad_logistica * 100, 2),
    round(especificidad_ridge * 100, 2),
    round(especificidad_arbol * 100, 2),
    round(especificidad_rf * 100, 2),
    round(especificidad_gbm * 100, 2),
    round(especificidad_xgb * 100, 2)
  ),
  Precisión = c(
    round(precision_logistica * 100, 2),
    round(precision_ridge * 100, 2),
    round(precision_arbol * 100, 2),
    round(precision_rf * 100, 2),
    round(precision_gbm * 100, 2),
    round(precision_xgb * 100, 2)
  ),
  "F1-Score" = c(
    round(f1_logistica, 4),
    round(f1_ridge, 4),
    round(f1_arbol, 4),
    round(f1_rf, 4),
    round(f1_gbm, 4),
    round(f1_xgb, 4)
  ),
  AUC = c(
    round(auc_logistica, 4),
    round(auc_ridge, 4),
    round(auc_arbol, 4),
    round(auc_rf, 4),
    round(auc_gbm, 4),
    round(auc_xgb, 4)
  ),
  stringsAsFactors = FALSE
)

print(comparacion)

cat("\n--- Ranking por AUC (todos los modelos) ---\n")
ranking_auc <- data.frame(
  Modelo = c("Logística", "Ridge", "Random Forest", "GBM", "XGBoost", "Árbol (RPART)"),
  AUC = c(auc_logistica, auc_ridge, auc_rf, auc_gbm, auc_xgb, auc_arbol)
)
ranking_auc <- ranking_auc[order(ranking_auc$AUC, decreasing = TRUE), ]
rownames(ranking_auc) <- 1:nrow(ranking_auc)
print(ranking_auc)

cat("\n" , sep = "")
cat("Modelo con mejor AUC: ", ranking_auc$Modelo[1], " (", round(ranking_auc$AUC[1], 4), ")\n", sep = "")
cat("\n=== FIN DEL ANÁLISIS ===\n")

