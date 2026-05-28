# ============================================================================
# BENCHMARK: REGRESIÓN LOGÍSTICA CON DATOS GRANDES
# 500,000 registros para entrenamiento + 125,000 para validación
# 
# Objetivo: Medir tiempo de ejecución de la regresión logística
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
library(data.table)

# ============================================================================
# ETAPA 1: CARGA Y PREPARACIÓN INICIAL DE DATOS
# ============================================================================

cat("\n=== ETAPA 1: CARGA Y PREPARACIÓN DE DATOS ===\n")

# Lectura de datos
data <- read_csv("airline_2m.csv", show_col_types = FALSE)
setDT(data)

# Definición de variables prevuelo
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

cat("Datos iniciales: ", nrow(data_modelo), " filas\n", sep = "")

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

cat("Datos después de limpieza: ", nrow(data_final), " filas\n", sep = "")

# ============================================================================
# ETAPA 2: CREACIÓN DE VARIABLES INTERPRETABLES
# ============================================================================

cat("\n=== ETAPA 2: CREACIÓN DE VARIABLES INTERPRETABLES ===\n")

# Función para categorizar riesgo basado en tasa de retrasos
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

# Función para categorizar hora del día
categorizar_hora <- function(hora) {
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

# Categorizar y crear tabla interpretable
riesgo_origen <- categorizar_riesgo(data_final, "OriginAirportID", "ArrDel15")
riesgo_destino <- categorizar_riesgo(data_final, "DestAirportID", "ArrDel15")
riesgo_aerolinea <- categorizar_riesgo(data_final, "Reporting_Airline", "ArrDel15")
riesgo_distancia <- categorizar_riesgo(data_final, "DistanceGroup", "ArrDel15")

tiempo_temp <- data_final %>%
  mutate(TiempoGrupo = factor(categorizar_tiempo_grupo(CRSElapsedTime)))
riesgo_tiempo <- categorizar_riesgo(tiempo_temp, "TiempoGrupo", "ArrDel15")

# Crear tabla con variables interpretables
data_interpretable <- data_final %>%
  mutate(TiempoGrupo = factor(categorizar_tiempo_grupo(CRSElapsedTime))) %>%
  left_join(riesgo_origen %>% select(OriginAirportID, riesgo),
            by = "OriginAirportID") %>%
  rename(OriginRiesgo = riesgo) %>%
  left_join(riesgo_destino %>% select(DestAirportID, riesgo),
            by = "DestAirportID") %>%
  rename(DestRiesgo = riesgo) %>%
  left_join(riesgo_aerolinea %>% select(Reporting_Airline, riesgo),
            by = "Reporting_Airline") %>%
  rename(AerolineaRiesgo = riesgo) %>%
  left_join(riesgo_distancia %>% select(DistanceGroup, riesgo),
            by = "DistanceGroup") %>%
  rename(DistanciaRiesgo = riesgo) %>%
  left_join(riesgo_tiempo %>% select(TiempoGrupo, riesgo),
            by = "TiempoGrupo") %>%
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

cat("✓ Variables interpretables creadas\n")

# ============================================================================
# ETAPA 3: PREPARACIÓN DE DATOS PARA BENCHMARK
# ============================================================================

cat("\n=== ETAPA 3: PARTICIÓN DE DATOS ===\n")

set.seed(2026)

# Usar toda la base de datos disponible
data_muestra <- data_interpretable

cat("Total de registros disponibles: ", nrow(data_muestra), "\n", sep = "")

# Partición estratificada: 75% entrenamiento, 25% validación (1/4)
trainIndex <- createDataPartition(
  data_muestra$ArrDel15,
  p = 0.75,
  list = FALSE,
  times = 1
)

datos_train <- data_muestra[trainIndex, ]
datos_test <- data_muestra[-trainIndex, ]

cat("\nPartición realizada:\n")
cat("  - Entrenamiento: ", nrow(datos_train), " registros (", 
    round(nrow(datos_train) / nrow(data_muestra) * 100, 1), "%)\n", sep = "")
cat("  - Validación: ", nrow(datos_test), " registros (", 
    round(nrow(datos_test) / nrow(data_muestra) * 100, 1), "%)\n", sep = "")

cat("\nProporción de retrasos en entrenamiento:\n")
print(prop.table(table(datos_train$ArrDel15)))

# ============================================================================
# ETAPA 4: PREPARACIÓN DE MATRICES PARA MODELADO
# ============================================================================

cat("\n=== ETAPA 4: PREPARACIÓN DE MATRICES DISPERSAS ===\n")

cat("Creando matrices dispersas...\n")
tiempo_matrices <- system.time({
  # Combinar para matriz modelo consistente
  datos_temp <- bind_rows(datos_train, datos_test)
  X_temp <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_temp)
  y_temp <- as.numeric(datos_temp$ArrDel15) - 1
  
  n_train_actual <- nrow(datos_train)
  n_test_actual <- nrow(datos_test)
  
  X_train <- X_temp[1:n_train_actual, ]
  y_train <- y_temp[1:n_train_actual]
  X_test <- X_temp[(n_train_actual + 1):(n_train_actual + n_test_actual), ]
  y_test <- y_temp[(n_train_actual + 1):(n_train_actual + n_test_actual)]
})

cat("✓ Matrices creadas\n")
cat("  - Tiempo de creación: ", round(tiempo_matrices[3], 2), " segundos\n", sep = "")
cat("  - Matriz X_train: ", nrow(X_train), " x ", ncol(X_train), "\n", sep = "")
cat("  - Matriz X_test: ", nrow(X_test), " x ", ncol(X_test), "\n", sep = "")

# Calcular pesos de clase
n_total <- length(y_train)
n_class_0 <- sum(y_train == 0)
n_class_1 <- sum(y_train == 1)

weight_0 <- n_total / (2 * n_class_0)
weight_1 <- n_total / (2 * n_class_1)
pesos <- ifelse(y_train == 0, weight_0, weight_1)

cat("\nPesos de clase:\n")
cat("  Clase 0 (No): ", round(weight_0, 3), "\n", sep = "")
cat("  Clase 1 (Si): ", round(weight_1, 3), "\n", sep = "")

# ============================================================================
# ETAPA 5: BENCHMARK - REGRESIÓN LOGÍSTICA
# ============================================================================

cat("\n\n" , sep = "")
cat("════════════════════════════════════════════════════════════════════════════════\n")
cat("                     BENCHMARK: REGRESIÓN LOGÍSTICA                             \n")
cat("════════════════════════════════════════════════════════════════════════════════\n")

cat("\n--- ENTRENAMIENTO DEL MODELO LOGÍSTICO ---\n")

set.seed(2026)

# Medir tiempo de entrenamiento
tiempo_entrenamiento <- system.time({
  modelo_logistico <- glmnet(
    x = X_train,
    y = y_train,
    weights = pesos,
    family = "binomial",
    standardize = TRUE,
    lambda = 0
  )
})

cat("\n✓ Modelo logístico entrenado\n")
cat("  Tiempo de entrenamiento: ", round(tiempo_entrenamiento[3], 2), " segundos\n", sep = "")
cat("  Tiempo de entrenamiento: ", round(tiempo_entrenamiento[3] / 60, 2), " minutos\n", sep = "")

# Medir tiempo de predicción
cat("\n--- PREDICCIÓN EN DATOS DE VALIDACIÓN ---\n")

tiempo_prediccion <- system.time({
  prediccion_logistica <- predict(modelo_logistico, 
                                  newx = X_test, type = "response")
})

cat("✓ Predicciones generadas\n")
cat("  Tiempo de predicción: ", round(tiempo_prediccion[3], 2), " segundos\n", sep = "")
cat("  Registros por segundo: ", round(nrow(X_test) / tiempo_prediccion[3], 0), "\n", sep = "")

# Medir tiempo de evaluación
cat("\n--- EVALUACIÓN DEL MODELO ---\n")

tiempo_evaluacion <- system.time({
  # Conversión a clases
  predicciones_clase <- ifelse(prediccion_logistica > 0.5, 1, 0)
  pred_factor <- factor(predicciones_clase, levels = c(0, 1))
  y_test_factor <- factor(y_test, levels = c(0, 1))
  
  # Matriz de confusión
  matriz_confusion <- confusionMatrix(data = pred_factor, 
                                      reference = y_test_factor, 
                                      positive = "1")
  
  # Cálculo de métricas
  tp <- matriz_confusion$table[2, 2]
  fn <- matriz_confusion$table[1, 2]
  fp <- matriz_confusion$table[2, 1]
  tn <- matriz_confusion$table[1, 1]
  
  sensibilidad <- tp / (tp + fn)
  especificidad <- tn / (tn + fp)
  precision <- tp / (tp + fp)
  f1 <- 2 * (precision * sensibilidad) / (precision + sensibilidad)
  accuracy <- (tn + tp) / (tn + tp + fp + fn)
  
  # AUC
  curva_roc <- roc(response = y_test, predictor = as.numeric(prediccion_logistica))
  auc_val <- auc(curva_roc)[1]
})

cat("✓ Evaluación completada\n")
cat("  Tiempo de evaluación: ", round(tiempo_evaluacion[3], 2), " segundos\n", sep = "")

# ============================================================================
# ETAPA 6: RESULTADOS DEL BENCHMARK
# ============================================================================

cat("\n\n" , sep = "")
cat("════════════════════════════════════════════════════════════════════════════════\n")
cat("                        RESULTADOS DEL BENCHMARK                               \n")
cat("════════════════════════════════════════════════════════════════════════════════\n")

# Tiempo total
tiempo_total <- tiempo_entrenamiento[3] + tiempo_prediccion[3] + tiempo_evaluacion[3]

cat("\n--- RESUMEN DE TIEMPOS ---\n")
cat("Entrenamiento: ", sprintf("%10.2f", tiempo_entrenamiento[3]), " segundos (", 
    sprintf("%6.2f", tiempo_entrenamiento[3] / 60), " minutos)\n", sep = "")
cat("Predicción:    ", sprintf("%10.2f", tiempo_prediccion[3]), " segundos\n", sep = "")
cat("Evaluación:    ", sprintf("%10.2f", tiempo_evaluacion[3]), " segundos\n", sep = "")
cat("───────────────────────────────────────────────────────────────────\n")
cat("TIEMPO TOTAL:  ", sprintf("%10.2f", tiempo_total), " segundos (", 
    sprintf("%6.2f", tiempo_total / 60), " minutos)\n", sep = "")

cat("\n--- DESEMPEÑO DEL MODELO ---\n")
cat("Accuracy:      ", sprintf("%6.2f", accuracy * 100), " %\n", sep = "")
cat("Sensibilidad:  ", sprintf("%6.2f", sensibilidad * 100), " %\n", sep = "")
cat("Especificidad: ", sprintf("%6.2f", especificidad * 100), " %\n", sep = "")
cat("Precisión:     ", sprintf("%6.2f", precision * 100), " %\n", sep = "")
cat("F1-Score:      ", sprintf("%6.4f", f1), "\n", sep = "")
cat("AUC:           ", sprintf("%6.4f", auc_val), "\n", sep = "")

cat("\n--- INFORMACIÓN DEL DATASET ---\n")
cat("Registros de entrenamiento: ", nrow(X_train), "\n", sep = "")
cat("Registros de validación:    ", nrow(X_test), "\n", sep = "")
cat("Total de variables:         ", ncol(X_train), "\n", sep = "")
cat("Proporción positivos (train):", sprintf("%6.2f", sum(y_train) / length(y_train) * 100), " %\n", sep = "")

# Matriz de confusión
cat("\n--- MATRIZ DE CONFUSIÓN ---\n")
print(matriz_confusion$table)

# ============================================================================
# ETAPA 7: GUARDADO DE RESULTADOS
# ============================================================================

cat("\n\n" , sep = "")
cat("════════════════════════════════════════════════════════════════════════════════\n")
cat("                      GUARDANDO RESULTADOS                                      \n")
cat("════════════════════════════════════════════════════════════════════════════════\n")

# Crear archivo de resumen
resumen_benchmark <- data.frame(
  Métrica = c(
    "Registros Entrenamiento",
    "Registros Validación",
    "Total Registros",
    "Variables",
    "Proporción Positivos (%)",
    "Tiempo Entrenamiento (s)",
    "Tiempo Predicción (s)",
    "Tiempo Evaluación (s)",
    "Tiempo Total (s)",
    "Tiempo Total (min)",
    "Accuracy (%)",
    "Sensibilidad (%)",
    "Especificidad (%)",
    "Precisión (%)",
    "F1-Score",
    "AUC"
  ),
  Valor = c(
    nrow(X_train),
    nrow(X_test),
    nrow(X_train) + nrow(X_test),
    ncol(X_train),
    round(sum(y_train) / length(y_train) * 100, 2),
    round(tiempo_entrenamiento[3], 2),
    round(tiempo_prediccion[3], 2),
    round(tiempo_evaluacion[3], 2),
    round(tiempo_total, 2),
    round(tiempo_total / 60, 2),
    round(accuracy * 100, 2),
    round(sensibilidad * 100, 2),
    round(especificidad * 100, 2),
    round(precision * 100, 2),
    round(f1, 4),
    round(auc_val, 4)
  ),
  stringsAsFactors = FALSE
)

# Guardar resumen
write.csv(resumen_benchmark, "resultados_analisis/benchmark_regresion_logistica.csv", 
          row.names = FALSE)
cat("\n✓ Guardado: resultados_analisis/benchmark_regresion_logistica.csv\n")

# Guardar reporte completo de texto
reporte <- capture.output({
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  cat("                  REPORTE BENCHMARK: REGRESIÓN LOGÍSTICA                       \n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  cat("--- CONFIGURACIÓN ---\n")
  cat("Registros entrenamiento: ", nrow(X_train), "\n", sep = "")
  cat("Registros validación:    ", nrow(X_test), "\n", sep = "")
  cat("Total de variables:      ", ncol(X_train), "\n", sep = "")
  cat("Proporción positivos:    ", round(sum(y_train) / length(y_train) * 100, 2), "%\n\n", sep = "")
  
  cat("--- TIEMPOS DE EJECUCIÓN ---\n")
  cat("Entrenamiento: ", round(tiempo_entrenamiento[3], 2), " segundos\n", sep = "")
  cat("Predicción:    ", round(tiempo_prediccion[3], 2), " segundos\n", sep = "")
  cat("Evaluación:    ", round(tiempo_evaluacion[3], 2), " segundos\n", sep = "")
  cat("TOTAL:         ", round(tiempo_total, 2), " segundos (", 
      round(tiempo_total / 60, 2), " minutos)\n\n", sep = "")
  
  cat("--- DESEMPEÑO ---\n")
  cat("Accuracy:      ", round(accuracy * 100, 2), "%\n", sep = "")
  cat("Sensibilidad:  ", round(sensibilidad * 100, 2), "%\n", sep = "")
  cat("Especificidad: ", round(especificidad * 100, 2), "%\n", sep = "")
  cat("Precisión:     ", round(precision * 100, 2), "%\n", sep = "")
  cat("F1-Score:      ", round(f1, 4), "\n", sep = "")
  cat("AUC:           ", round(auc_val, 4), "\n")
})

writeLines(reporte, "resultados_analisis/benchmark_reporte_regresion_logistica.txt")
cat("✓ Guardado: resultados_analisis/benchmark_reporte_regresion_logistica.txt\n")

cat("\n════════════════════════════════════════════════════════════════════════════════\n")
cat("                        BENCHMARK COMPLETADO                                    \n")
cat("════════════════════════════════════════════════════════════════════════════════\n\n")
