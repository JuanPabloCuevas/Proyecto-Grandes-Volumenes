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
# Etapa 10: Estadísticas descriptivas y visualizaciones de modelos
# Etapa 11: Comparación de todos los modelos
# ============================================================================


# ============================================================================
# LIBRERÍAS
# ============================================================================
library(dplyr)
library(readr)
library(tidyr)
library(tibble)
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
library(ggplot2)
library(gridExtra)
library(gt)

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
data_muestra <- data_interpretable %>% slice_sample(n = n_muestra_total)

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
# ETAPA 5B: MODELO LASSO
# ============================================================================

cat("\n=== MODELO LASSO ===\n")

# Entrenar Lasso
modelo_lasso <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 1,
  standardize = TRUE
)

# Validación cruzada para obtener lambda óptimo
cv_modelo_lasso <- cv.glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 1,
  nfolds = 5
)

lambda_optimo_lasso <- cv_modelo_lasso$lambda.min

# Predicciones con lasso usando lambda óptimo
prediccion_lasso <- predict(modelo_lasso, 
                            newx = X_test, type = "response", s = lambda_optimo_lasso)

# Convertir a clases
predicciones_clase_lasso <- ifelse(prediccion_lasso > 0.5, 1, 0)
pred_factor_lasso <- factor(predicciones_clase_lasso, levels = c(0, 1))

# Matriz de confusión
matriz_confusion_lasso <- confusionMatrix(data = pred_factor_lasso, 
                                          reference = y_test_factor, 
                                          positive = "1")

cat("\n=== EVALUACIÓN - LASSO ===\n")
print(matriz_confusion_lasso)

# Calcular Sensibilidad (Recall), Especificidad y Precisión
tp_lasso <- matriz_confusion_lasso$table[2, 2]
fn_lasso <- matriz_confusion_lasso$table[1, 2]
fp_lasso <- matriz_confusion_lasso$table[2, 1]
tn_lasso <- matriz_confusion_lasso$table[1, 1]
sensibilidad_lasso <- tp_lasso / (tp_lasso + fn_lasso)
especificidad_lasso <- tn_lasso / (tn_lasso + fp_lasso)
precision_lasso <- tp_lasso / (tp_lasso + fp_lasso)
f1_lasso <- 2 * (precision_lasso * sensibilidad_lasso) / (precision_lasso + sensibilidad_lasso)

# AUC
curva_roc_lasso <- roc(response = y_test, predictor = as.numeric(prediccion_lasso))
auc_lasso <- auc(curva_roc_lasso)[1]

cat("\nSensibilidad (Recall): ", round(sensibilidad_lasso * 100, 2), "%\n", sep = "")
cat("Especificidad: ", round(especificidad_lasso * 100, 2), "%\n", sep = "")
cat("Precisión: ", round(precision_lasso * 100, 2), "%\n", sep = "")
cat("F1-Score: ", round(f1_lasso, 4), "\n", sep = "")
cat("AUC: ", round(auc_lasso, 4), "\n", sep = "")

cat("\n✓ Modelo Lasso entrenado\n")
cat("  Lambda óptimo: ", round(lambda_optimo_lasso, 6), "\n", sep = "")

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

# Visualizar con estilo mejorado
cat("\nGenerando visualización del árbol...\n")

# Opción 1: rpart.plot mejorado con mejor estilo
png("resultados_analisis/arbol_clasificacion.png", width = 1400, height = 900, res = 120)
rpart.plot(
  arbol_pruned,
  main = "Árbol de Clasificación - Retrasos Aéreos",
  type = 1,                    # Tipo de árbol
  extra = 101,                 # Mostrar probabilidades
  cex = 0.8,                   # Tamaño del texto
  fallen.leaves = FALSE,       # Hojas en el fondo
  box.palette = "RdYlGn",      # Paleta de colores
  shadow.col = "gray70",       # Sombra para profundidad
  tweak = 0.8,                 # Ajuste del espaciado
  compress = FALSE             # No comprimir el árbol
)
dev.off()
cat("✓ Árbol guardado en: resultados_analisis/arbol_clasificacion.png\n")

# Visualizar en pantalla
rpart.plot(
  arbol_pruned,
  main = "Árbol de Clasificación - Retrasos Aéreos",
  type = 1,                    # Tipo de árbol
  extra = 101,                 # Mostrar probabilidades
  cex = 0.8,                   # Tamaño del texto
  fallen.leaves = FALSE,       # Hojas en el fondo
  box.palette = "RdYlGn",      # Paleta de colores
  shadow.col = "gray70",       # Sombra para profundidad
  tweak = 0.8,                 # Ajuste del espaciado
  compress = FALSE             # No comprimir el árbol
)

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
# ETAPA 10: ESTADÍSTICAS DESCRIPTIVAS Y VISUALIZACIONES
# ============================================================================

cat("\n\n=== ETAPA 10: ESTADÍSTICAS DESCRIPTIVAS Y VISUALIZACIONES ===")

# Función auxiliar para crear matriz de confusión con ggplot
graficar_confusion_matrix <- function(cm, titulo) {
  # Convertir tabla a data frame
  cm_df <- as.data.frame(as.table(cm))
  colnames(cm_df) <- c("Predicho", "Real", "Freq")
  
  # Crear gráfico
  ggplot(cm_df, aes(x = Predicho, y = Real, fill = Freq)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = Freq), vjust = 0.5, size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#cffafe", high = "#0c4a6e") +
    labs(title = titulo,
         x = "Predicho",
         y = "Real",
         fill = "Frecuencia") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          axis.text = element_text(size = 10, face = "bold"),
          axis.title = element_text(size = 11, face = "bold"),
          panel.grid = element_blank(),
          aspect.ratio = 1)
}

cat("\n--- Matrices de Confusión por Modelo ---\n")

# Crear gráficos de matrices de confusión
plot_logistica <- graficar_confusion_matrix(matriz_confusion$table, "Regresión Logística")
plot_ridge <- graficar_confusion_matrix(matriz_confusion_ridge$table, "Ridge")
plot_lasso <- graficar_confusion_matrix(matriz_confusion_lasso$table, "Lasso")
plot_arbol <- graficar_confusion_matrix(confusion_matrix, "Árbol de Clasificación")
plot_rf <- graficar_confusion_matrix(rf_cm, "Random Forest")
plot_gbm <- graficar_confusion_matrix(gbm_cm, "GBM")
plot_xgb <- graficar_confusion_matrix(xgb_cm, "XGBoost")

# Mostrar matrices de confusión en una grilla
grilla_confusion <- gridExtra::grid.arrange(
  plot_logistica, plot_ridge, plot_lasso, plot_arbol,
  plot_rf, plot_gbm, plot_xgb,
  ncol = 4
)
print(grilla_confusion)

cat("\n--- Curvas ROC Comparativas ---\n")

# Crear data frame con todas las ROC
roc_data <- data.frame(
  FPR = numeric(),
  TPR = numeric(),
  Modelo = character(),
  AUC = numeric()
)

# Agregar datos de cada modelo
modelos_roc <- list(
  list(roc = curva_roc, nombre = "Logística", auc = auc_logistica),
  list(roc = curva_roc_ridge, nombre = "Ridge", auc = auc_ridge),
  list(roc = curva_roc_lasso, nombre = "Lasso", auc = auc_lasso),
  list(roc = roc_arbol, nombre = "Árbol", auc = auc_arbol),
  list(roc = roc_rf, nombre = "Random Forest", auc = auc_rf),
  list(roc = roc_gbm, nombre = "GBM", auc = auc_gbm),
  list(roc = roc_xgb, nombre = "XGBoost", auc = auc_xgb)
)

for (modelo in modelos_roc) {
  coords <- coords(modelo$roc, "all", ret = c("specificity", "sensitivity"))
  temp_df <- data.frame(
    FPR = 1 - coords[, 1],
    TPR = coords[, 2],
    Modelo = modelo$nombre,
    AUC = modelo$auc
  )
  roc_data <- rbind(roc_data, temp_df)
}

# Modelos Tree-based (Árbol, Random Forest, GBM, XGBoost)
roc_arboles <- roc_data %>% filter(Modelo %in% c("Árbol", "Random Forest", "GBM", "XGBoost"))

plot_roc_arboles <- ggplot(roc_arboles, aes(x = FPR, y = TPR, color = Modelo, linetype = Modelo)) +
  geom_line(size = 1) +
  geom_abline(intercept = 0, slope = 1, color = "gray", linetype = "dashed", size = 1) +
  scale_color_manual(values = c("Árbol" = "#22C55E", "Random Forest" = "#F97316",
                                 "GBM" = "#A855F7", "XGBoost" = "#92400E")) +
  scale_linetype_manual(values = c("Árbol" = 1, "Random Forest" = 1, "GBM" = 1, "XGBoost" = 1)) +
  labs(title = "Curvas ROC - Modelos Tree-based",
       x = "Tasa de Falsos Positivos (FPR)",
       y = "Tasa de Verdaderos Positivos (TPR)",
       color = "Modelo",
       linetype = "Modelo") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 11, face = "bold"),
        legend.position = "bottom",
        panel.grid.major = element_line(color = "#ecf0f1"))

print(plot_roc_arboles)

cat("\n--- Tablas de Variables Importantes (gt) ---\n")

# Tabla de importancia - Random Forest
cat("\n• Random Forest - Top 10 Variables\n")
rf_imp_df <- data.frame(
  Variable = names(rf_model$variable.importance[1:10]),
  Importancia = as.numeric(rf_model$variable.importance[1:10])
) %>%
  mutate(Importancia_Rel = round((Importancia / max(Importancia)) * 100, 2),
         Ranking = row_number())

rf_gt <- rf_imp_df %>%
  select(Ranking, Variable, Importancia_Rel) %>%
  gt() %>%
  cols_label(
    Ranking = "Ranking",
    Variable = "Variable",
    Importancia_Rel = "Importancia Relativa (%)"
  ) %>%
  tab_header(
    title = "Random Forest",
    subtitle = "Variables Más Importantes"
  ) %>%
  fmt_number(columns = Importancia_Rel, decimals = 2) %>%
  opt_table_font(font = "helvetica") %>%
  tab_style(
    style = cell_fill(color = "#e0f2fe"),
    locations = cells_body(rows = seq(1, nrow(rf_imp_df), 2))
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  )

print(rf_gt)

# Tabla de importancia - GBM
cat("\n• GBM - Top 10 Variables\n")
gbm_imp <- summary(gbm_model, n.trees = 500, plotit = FALSE)
gbm_imp_df <- gbm_imp[1:10, ] %>%
  rownames_to_column(var = "Variable") %>%
  mutate(Importancia_Rel = round((rel.inf / max(rel.inf)) * 100, 2),
         Ranking = row_number()) %>%
  select(Ranking, Variable, Importancia_Rel)

gbm_gt <- gbm_imp_df %>%
  gt() %>%
  cols_label(
    Ranking = "Ranking",
    Variable = "Variable",
    Importancia_Rel = "Importancia Relativa (%)"
  ) %>%
  tab_header(
    title = "Gradient Boosting (GBM)",
    subtitle = "Variables Más Importantes"
  ) %>%
  fmt_number(columns = Importancia_Rel, decimals = 2) %>%
  opt_table_font(font = "helvetica") %>%
  tab_style(
    style = cell_fill(color = "#e0f2fe"),
    locations = cells_body(rows = seq(1, nrow(gbm_imp_df), 2))
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  )

print(gbm_gt)

# Tabla de importancia - XGBoost
cat("\n• XGBoost - Top 10 Variables\n")
xgb_imp_df <- xgb_importance[1:10, ] %>%
  as.data.frame() %>%
  mutate(Importancia_Rel = round((Gain / max(Gain)) * 100, 2),
         Ranking = row_number()) %>%
  select(Ranking, Feature, Importancia_Rel) %>%
  rename(Variable = Feature)

xgb_gt <- xgb_imp_df %>%
  gt() %>%
  cols_label(
    Ranking = "Ranking",
    Variable = "Variable",
    Importancia_Rel = "Importancia Relativa (%)"
  ) %>%
  tab_header(
    title = "XGBoost",
    subtitle = "Variables Más Importantes"
  ) %>%
  fmt_number(columns = Importancia_Rel, decimals = 2) %>%
  opt_table_font(font = "helvetica") %>%
  tab_style(
    style = cell_fill(color = "#e0f2fe"),
    locations = cells_body(rows = seq(1, nrow(xgb_imp_df), 2))
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  )

print(xgb_gt)

cat("\n✓ Estadísticas descriptivas y visualizaciones completadas\n")

# ============================================================================
# ETAPA 11: COMPARACIÓN DE TODOS LOS MODELOS
# ============================================================================

cat("\n\n=== ETAPA 11: COMPARACIÓN DE MODELOS ===")
cat("\n--- Resumen de desempeño de todos los modelos ---\n")

# Calcular Accuracy para cada modelo
accuracy_logistica <- (tn_logistica + tp_logistica) / (tn_logistica + tp_logistica + fp_logistica + fn_logistica)
accuracy_ridge <- (tn_ridge + tp_ridge) / (tn_ridge + tp_ridge + fp_ridge + fn_ridge)
accuracy_lasso <- (tn_lasso + tp_lasso) / (tn_lasso + tp_lasso + fp_lasso + fn_lasso)
accuracy_arbol <- (tn_arbol + tp_arbol) / (tn_arbol + tp_arbol + fp_arbol + fn_arbol)
accuracy_rf <- (tn_rf + tp_rf) / (tn_rf + tp_rf + fp_rf + fn_rf)
accuracy_gbm <- (tn_gbm + tp_gbm) / (tn_gbm + tp_gbm + fp_gbm + fn_gbm)
accuracy_xgb <- (tn_xgb + tp_xgb) / (tn_xgb + tp_xgb + fp_xgb + fn_xgb)

# Crear tabla comparativa
comparacion <- data.frame(
  Modelo = c("Logística", "Ridge", "Lasso", "Árbol (RPART)", "Random Forest", "GBM", "XGBoost"),
  Accuracy = c(
    round(accuracy_logistica * 100, 2),
    round(accuracy_ridge * 100, 2),
    round(accuracy_lasso * 100, 2),
    round(accuracy_arbol * 100, 2),
    round(accuracy_rf * 100, 2),
    round(accuracy_gbm * 100, 2),
    round(accuracy_xgb * 100, 2)
  ),
  Sensibilidad = c(
    round(sensibilidad_logistica * 100, 2),
    round(sensibilidad_ridge * 100, 2),
    round(sensibilidad_lasso * 100, 2),
    round(sensibilidad_arbol * 100, 2),
    round(sensibilidad_rf * 100, 2),
    round(sensibilidad_gbm * 100, 2),
    round(sensibilidad_xgb * 100, 2)
  ),
  Especificidad = c(
    round(especificidad_logistica * 100, 2),
    round(especificidad_ridge * 100, 2),
    round(especificidad_lasso * 100, 2),
    round(especificidad_arbol * 100, 2),
    round(especificidad_rf * 100, 2),
    round(especificidad_gbm * 100, 2),
    round(especificidad_xgb * 100, 2)
  ),
  Precisión = c(
    round(precision_logistica * 100, 2),
    round(precision_ridge * 100, 2),
    round(precision_lasso * 100, 2),
    round(precision_arbol * 100, 2),
    round(precision_rf * 100, 2),
    round(precision_gbm * 100, 2),
    round(precision_xgb * 100, 2)
  ),
  `F1-Score` = c(
    round(f1_logistica, 4),
    round(f1_ridge, 4),
    round(f1_lasso, 4),
    round(f1_arbol, 4),
    round(f1_rf, 4),
    round(f1_gbm, 4),
    round(f1_xgb, 4)
  ),
  AUC = c(
    round(auc_logistica, 4),
    round(auc_ridge, 4),
    round(auc_lasso, 4),
    round(auc_arbol, 4),
    round(auc_rf, 4),
    round(auc_gbm, 4),
    round(auc_xgb, 4)
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# Tabla gt - Resumen de desempeño
comparacion_gt <- comparacion %>%
  gt() %>%
  cols_label(
    Modelo = "Modelo",
    Accuracy = "Accuracy (%)",
    Sensibilidad = "Sensibilidad (%)",
    Especificidad = "Especificidad (%)",
    Precisión = "Precisión (%)",
    `F1-Score` = "F1-Score",
    AUC = "AUC"
  ) %>%
  tab_header(
    title = "Resumen de Desempeño",
    subtitle = "Comparación de todos los modelos"
  ) %>%
  fmt_number(columns = c(Accuracy, Sensibilidad, Especificidad, Precisión), decimals = 2) %>%
  fmt_number(columns = c(`F1-Score`, AUC), decimals = 4) %>%
  opt_table_font(font = "helvetica") %>%
  tab_style(
    style = cell_fill(color = "#e0f2fe"),
    locations = cells_body(rows = seq(1, nrow(comparacion), 2))
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(columns = Modelo)
  )

print(comparacion_gt)

cat("\n--- Ranking por AUC (todos los modelos) ---\n")
ranking_auc <- data.frame(
  Modelo = c("Logística", "Ridge", "Lasso", "Random Forest", "GBM", "XGBoost", "Árbol (RPART)"),
  AUC = c(auc_logistica, auc_ridge, auc_lasso, auc_rf, auc_gbm, auc_xgb, auc_arbol)
)
ranking_auc <- ranking_auc[order(ranking_auc$AUC, decreasing = TRUE), ]
ranking_auc <- ranking_auc %>%
  mutate(Ranking = row_number()) %>%
  select(Ranking, Modelo, AUC)

# Tabla gt - Ranking por AUC
ranking_gt <- ranking_auc %>%
  gt() %>%
  cols_label(
    Ranking = "Ranking",
    Modelo = "Modelo",
    AUC = "AUC"
  ) %>%
  tab_header(
    title = "Ranking de Modelos",
    subtitle = "Ordenados por AUC (Descendente)"
  ) %>%
  fmt_number(columns = AUC, decimals = 4) %>%
  opt_table_font(font = "helvetica") %>%
  tab_style(
    style = cell_fill(color = "#e0f2fe"),
    locations = cells_body(rows = seq(1, nrow(ranking_auc), 2))
  ) %>%
  tab_style(
    style = cell_fill(color = "#bae6fd"),
    locations = cells_body(rows = 1)
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_style(
    style = cell_text(weight = "bold", size = "large"),
    locations = cells_body(rows = 1)
  )

print(ranking_gt)

cat("\n" , sep = "")
cat("✓ Modelo con mejor AUC: ", ranking_auc$Modelo[1], " (", round(ranking_auc$AUC[1], 4), ")\n", sep = "")
cat("\n=== FIN DEL ANÁLISIS ===\n")

# ============================================================================
# DESCARGA DE GRÁFICOS Y TABLAS
# ============================================================================

cat("\n\n=== DESCARGANDO GRÁFICOS Y TABLAS ===\n")

# Crear directorio para guardar archivos si no existe
dir.create("resultados_analisis", showWarnings = FALSE)

# Guardar matrices de confusión individualmente
cat("\n--- Guardando matrices de confusión ---\n")
ggsave("resultados_analisis/01_matriz_confusion_logistica.png", plot_logistica, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 01_matriz_confusion_logistica.png\n")

ggsave("resultados_analisis/02_matriz_confusion_ridge.png", plot_ridge, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 02_matriz_confusion_ridge.png\n")

ggsave("resultados_analisis/03_matriz_confusion_lasso.png", plot_lasso, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 03_matriz_confusion_lasso.png\n")

ggsave("resultados_analisis/04_matriz_confusion_arbol.png", plot_arbol, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 04_matriz_confusion_arbol.png\n")

ggsave("resultados_analisis/05_matriz_confusion_randomforest.png", plot_rf, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 05_matriz_confusion_randomforest.png\n")

ggsave("resultados_analisis/06_matriz_confusion_gbm.png", plot_gbm, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 06_matriz_confusion_gbm.png\n")

ggsave("resultados_analisis/07_matriz_confusion_xgboost.png", plot_xgb, width = 8, height = 6, dpi = 300)
cat("✓ Guardado: 07_matriz_confusion_xgboost.png\n")

# Guardar gráfico ROC
cat("\n--- Guardando gráfico ROC ---\n")
ggsave("resultados_analisis/08_curvas_roc_tree_based.png", plot_roc_arboles, width = 10, height = 7, dpi = 300)
cat("✓ Guardado: 08_curvas_roc_tree_based.png\n")

# Configurar webshot2 para usar Brave Browser
cat("\n--- Configurando webshot2 para exportar tablas ---\n")
library(webshot2)
Sys.setenv(CHROMOTE_CHROME = "C:/Archivos de programa/BraveSoftware/Brave-Browser/Application/brave.exe")

# Guardar tablas gt
cat("\n--- Guardando tablas gt ---\n")
gtsave(comparacion_gt, "resultados_analisis/09_tabla_comparacion_modelos.png")
cat("✓ Guardado: 09_tabla_comparacion_modelos.png\n")

gtsave(ranking_gt, "resultados_analisis/10_tabla_ranking_auc.png")
cat("✓ Guardado: 10_tabla_ranking_auc.png\n")

gtsave(rf_gt, "resultados_analisis/11_tabla_importancia_randomforest.png")
cat("✓ Guardado: 11_tabla_importancia_randomforest.png\n")

gtsave(gbm_gt, "resultados_analisis/12_tabla_importancia_gbm.png")
cat("✓ Guardado: 12_tabla_importancia_gbm.png\n")

gtsave(xgb_gt, "resultados_analisis/13_tabla_importancia_xgboost.png")
cat("✓ Guardado: 13_tabla_importancia_xgboost.png\n")

cat("\n✓ Todos los archivos han sido guardados en la carpeta 'resultados_analisis'\n")
cat("=== DESCARGA COMPLETADA ===\n")

