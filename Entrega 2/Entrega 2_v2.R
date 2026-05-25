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
# ============================================================================

# ============================================================================
# LIBRERÍAS
# ============================================================================
library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(glmnet)
library(Matrix)
library(rpart)
library(rpart.plot)
library(data.table)

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


# Categorizar aeropuertos
riesgo_origen <- categorizar_riesgo(data_final, "OriginAirportID", "ArrDel15")
riesgo_destino <- categorizar_riesgo(data_final, "DestAirportID", "ArrDel15")

cat("  ✓ Origen: ", paste(table(riesgo_origen$riesgo), collapse = " | "), "\n", sep = "")
cat("  ✓ Destino: ", paste(table(riesgo_destino$riesgo), collapse = " | "), "\n", sep = "")

# Crear nueva tabla con variables interpretables
data_interpretable <- data_final %>%
  left_join(riesgo_origen %>% select(OriginAirportID, riesgo),
            by = "OriginAirportID", suffix = c("", "_origen")) %>%
  rename(OriginRiesgo = riesgo) %>%
  left_join(riesgo_destino %>% select(DestAirportID, riesgo),
            by = "DestAirportID", suffix = c("", "_destino")) %>%
  rename(DestRiesgo = riesgo) %>%
  mutate(
    OriginRiesgo = factor(OriginRiesgo, levels = c("Bajo", "Medio", "Alto")),
    DestRiesgo = factor(DestRiesgo, levels = c("Bajo", "Medio", "Alto")),
    HoraSalida = categorizar_hora(CRSDepTime),
    HoraLlegada = categorizar_hora(CRSArrTime),
    Estacion = factor(categorizar_estacion(Month), levels = c("Primavera", "Verano", "Otoño", "Invierno"))
  ) %>%
  select(-OriginAirportID, -DestAirportID, -CRSDepTime, -CRSArrTime, -Month, -OriginCityMarketID, -DestCityMarketID, -OriginWac, -DestWac)

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

# Partición estratificada
set.seed(2026)
trainIndex <- createDataPartition(
  data_interpretable$ArrDel15,
  p = 0.80,
  list = FALSE,
  times = 1
)

datos_train <- data_interpretable[trainIndex, ]
datos_test <- data_interpretable[-trainIndex, ]

cat("Entrenamiento (antes de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Prueba (antes de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# Sampling para manejo eficiente de memoria
n_train_sample <- min(100000, nrow(datos_train))
n_test_sample <- min(25000, nrow(datos_test))

datos_train <- datos_train %>% slice_sample(n = n_train_sample)
datos_test <- datos_test %>% slice_sample(n = n_test_sample)

cat("Entrenamiento (después de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Prueba (después de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# ============================================================================
#  MODELO LOGÍSTICO Y CON REGULARIZACIÓN RIDGE
# ============================================================================


cat("\n=== MODELO LOGÍSTICO===\n")

modelo_ridge <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  standardize = TRUE
)

# Preparar matriz para Ridge
datos_train_ridge <- datos_train
datos_test_ridge <- datos_test

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

# Entrenar Ridge
set.seed(2026)
modelo_ridge <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  standardize = TRUE
)

# Validación cruzada
set.seed(2026)
cv_modelo <- cv.glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,
  nfolds = 5
)

lambda_optimo <- cv_modelo$lambda.min
cat("✓ Modelo Ridge entrenado\n")
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
set.seed(2026)
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
rpart.plot(arbol_pruned, main = "Árbol de Clasificación - Retrasos Aéreos")

# Predicciones y evaluación
cat("\n--- Evaluación en datos de prueba ---\n")
predicciones_arbol <- predict(arbol_pruned, newdata = datos_test_rpart, type = "class")

confusion_matrix <- table(Predicho = predicciones_arbol, Real = datos_test_rpart$ArrDel15)
print(confusion_matrix)

exactitud <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\nExactitud del árbol: ", round(exactitud * 100, 2), "%\n", sep = "")

cat("\n=== FIN DEL ANÁLISIS ===\n")
