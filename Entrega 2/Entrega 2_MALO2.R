# ============================================================================
# ANÁLISIS PREDICTIVO: RETRASOS AÉREOS (ArrDel15)
# Etapa 1: Carga, selección y preparación de datos
# Etapa 2: Análisis descriptivo previo a modelado
# Etapa 3: Modelo logístico con regularización Ridge
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
library(data.table)


# ============================================================================
# ETAPA 1: CARGA Y PREPARACIÓN INICIAL DE DATOS
# ============================================================================

cat("\n=== ETAPA 1: CARGA Y PREPARACIÓN DE DATOS ===\n")

# Lectura de datos
data <- rio::import("airline_2m.csv")

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
  # antes de que deje la pista del aeropuerto de origen
  #,"DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
  #"TaxiOut", "WheelsOff"
)

# Selección inicial de variables y conversión de variable respuesta
data_modelo <- data %>%
  select(ArrDel15, any_of(variables_prevuelo)) %>%
  mutate(ArrDel15 = factor(ArrDel15, levels = c(0, 1), labels = c("No", "Si"))) %>%
  filter(!is.na(ArrDel15))

cat("Datos iniciales: ", nrow(data_modelo), " filas, ", ncol(data_modelo), " columnas\n", sep = "")

# Análisis de datos faltantes ANTES de eliminar
cat("\n=== ANÁLISIS DE DATOS FALTANTES (previo a limpieza) ===\n")
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

# Conversión de variables categóricas 
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

cat("\nDatos después de limpieza: ", nrow(data_final), " filas\n", ncol(data_final), " columnas\n",sep = "")

# ============================================================================
# ETAPA 2: ANÁLISIS DESCRIPTIVO PREVIO A MODELADO
# ============================================================================

cat("\n=== ETAPA 2: ANÁLISIS DESCRIPTIVO ===\n")

# 2.1. Distribución de variable respuesta
cat("\n--- Distribución de la variable respuesta ---\n")
class_dist <- data_final %>%
  count(ArrDel15) %>%
  mutate(Porcentaje = round(n / sum(n) * 100, 2))
print(class_dist)
prop_retrasado <- class_dist %>% filter(ArrDel15 == "Si") %>% pull(Porcentaje)

# 2.2. Correlación con variable respuesta y multicolinealidad
cat("\n--- Análisis de variables numéricas ---\n")

# Identificar variables numéricas en data_modelo
vars_numericas <- data_modelo %>% select(where(is.numeric)) %>% names()

if (length(vars_numericas) > 0) {
  # Correlación con ArrDel15
  cat("Correlaciones con ArrDel15:\n")
  ArrDel15_num <- as.numeric(data_modelo$ArrDel15) - 1
  
  correlaciones <- sapply(vars_numericas, function(var) {
    cor(data_modelo[[var]], ArrDel15_num, use = "complete.obs")
  })
  
  correlaciones_sorted <- sort(abs(correlaciones), decreasing = TRUE)
  for (i in seq_along(correlaciones_sorted)) {
    var_name <- names(correlaciones_sorted)[i]
    cor_value <- correlaciones[var_name]
    cat(sprintf("  %s: %.4f\n", var_name, cor_value))
  }
  
  # Multicolinealidad
  cat("\n--- Análisis de multicolinealidad (justificación para Ridge) ---\n")
  cor_matrix <- cor(data_modelo[, ..vars_numericas], use = "complete.obs")
  cor_altas <- which(abs(cor_matrix) > 0.7 & lower.tri(cor_matrix), arr.ind = TRUE)
  
  if (nrow(cor_altas) > 0) {
    cat("✓ Multicolinealidad detectada (correlaciones > 0.7):\n")
    for (i in 1:nrow(cor_altas)) {
      row_idx <- cor_altas[i, 1]
      col_idx <- cor_altas[i, 2]
      var1 <- vars_numericas[row_idx]
      var2 <- vars_numericas[col_idx]
      cor_value <- cor_matrix[row_idx, col_idx]
      cat(sprintf("  %s <-> %s: %.4f\n", var1, var2, cor_value))
    }
    cat("\n➜ Ridge es la opción adecuada para regularizar esta multicolinealidad.\n")
  } else {
    cat("⚠ Correlaciones moderadas entre predictoras.\n")
    cat("➜ Ridge proporciona regularización general para prevenir overfitting.\n")
  }
}

# ============================================================================
# ETAPA 3: PARTICIÓN Y PREPARACIÓN PARA MODELADO
# ============================================================================

cat("\n=== ETAPA 3: PREPARACIÓN DE DATOS PARA MODELADO ===\n")

# 3.1. Partición estratificada (mantiene proporción de clases)
set.seed(2026)
trainIndex <- createDataPartition(
  data_final$ArrDel15,
  p = 0.80,
  list = FALSE,
  times = 1
)

datos_train <- data_final[trainIndex, ]
datos_test <- data_final[-trainIndex, ]

cat("Entrenamiento (antes de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Prueba (antes de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# 3.2. Sampling para manejo eficiente de memoria
n_train_sample <- min(100000, nrow(datos_train))
n_test_sample <- min(25000, nrow(datos_test))

datos_train <- datos_train %>% slice_sample(n = n_train_sample)
datos_test <- datos_test %>% slice_sample(n = n_test_sample)

cat("Entrenamiento (después de sampling): ", nrow(datos_train), " filas\n", sep = "")
cat("Prueba (después de sampling):        ", nrow(datos_test), " filas\n", sep = "")

# ============================================================================
# ETAPA 4: TARGET ENCODING PARA VARIABLES DE ALTA CARDINALIDAD
# ============================================================================

cat("\n=== ETAPA 4: TARGET ENCODING ===")

# Función para aplicar target encoding
apply_target_encoding <- function(train_data, test_data, variable_name, target_name = "ArrDel15") {
  # Calcular el promedio de target para cada categoría en entrenamiento
  encodings <- train_data %>%
    group_by(!!sym(variable_name)) %>%
    summarise(
      target_encoded = mean(as.numeric(!!sym(target_name)) - 1, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Calcular el promedio global para rellenar valores desconocidos en prueba
  global_mean <- mean(as.numeric(train_data[[target_name]]) - 1, na.rm = TRUE)
  
  new_col_name <- paste0(variable_name, "_encoded")
  
  # Aplicar a datos de entrenamiento
  train_encoded <- train_data %>%
    left_join(encodings, by = variable_name) %>%
    mutate(!!sym(new_col_name) := coalesce(target_encoded, global_mean)) %>%
    select(-target_encoded)
  
  # Aplicar a datos de prueba
  test_encoded <- test_data %>%
    left_join(encodings, by = variable_name) %>%
    mutate(!!sym(new_col_name) := coalesce(target_encoded, global_mean)) %>%
    select(-target_encoded)
  
  return(list(train = train_encoded, test = test_encoded, encodings = encodings))
}

# Variables de alta cardinalidad a codificar
vars_alta_cardinalidad <- c("OriginAirportID", "DestAirportID", "OriginCityMarketID", "DestCityMarketID")

cat("\nAplicando Target Encoding a: ", paste(vars_alta_cardinalidad, collapse = ", "), "\n", sep = "")

# Aplicar target encoding a cada variable
for (var in vars_alta_cardinalidad) {
  if (var %in% names(datos_train)) {
    result <- apply_target_encoding(datos_train, datos_test, var)
    datos_train <- result$train
    datos_test <- result$test
    n_unique <- nrow(result$encodings)
    cat("  ✓ ", var, ": ", n_unique, " categorías -> 1 variable numérica\n", sep = "")
  }
}

cat("\n✓ Target Encoding completado\n")

# ============================================================================
# ETAPA 5: CONSTRUCCIÓN DE MATRICES PARA MODELADO
# ============================================================================

cat("\n=== ETAPA 5: CONSTRUCCIÓN DE MATRICES DISPERSAS ===")

# Variables a usar en el modelo
# Nota: Las variables de aeropuertos y mercados han sido convertidas a numéricas
#       mediante Target Encoding (promedio de retrasos por categoría).
#       Esto reduce de ~3000 columnas dummy a 4 variables numéricas interpretables.
variables_modelo <- c(
  "ArrDel15", "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
  "Reporting_Airline", "OriginState", "DestState",
  "CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
  "CRSElapsedTime", "Distance", "DistanceGroup",
  "OriginAirportID_encoded", "DestAirportID_encoded",
  "OriginCityMarketID_encoded", "DestCityMarketID_encoded"
)

# Combinar datos para crear matrices con columnas consistentes
datos_temp <- bind_rows(
  datos_train %>% select(all_of(variables_modelo)),
  datos_test %>% select(all_of(variables_modelo))
)

# Crear matriz dispersa
X_temp <- sparse.model.matrix(ArrDel15 ~ . - 1, data = datos_temp)
y_temp <- as.numeric(datos_temp$ArrDel15) - 1

# Dividir en entrenamiento y prueba
n_train <- nrow(datos_train)
n_test <- nrow(datos_test)

X_train <- X_temp[1:n_train, ]
y_train <- y_temp[1:n_train]

X_test <- X_temp[(n_train + 1):(n_train + n_test), ]
y_test <- y_temp[(n_train + 1):(n_train + n_test)]

cat("Matriz dispersa - Entrenamiento: ", nrow(X_train), " x ", ncol(X_train), "\n", sep = "")
cat("Matriz dispersa - Prueba:        ", nrow(X_test), " x ", ncol(X_test), "\n", sep = "")

if (ncol(X_train) == ncol(X_test)) {
  cat("✓ Las matrices tienen dimensiones consistentes\n")
} else {
  cat("⚠ ADVERTENCIA: Dimensiones inconsistentes\n")
}

# ============================================================================
# ETAPA 6: MODELO LOGÍSTICO CON REGULARIZACIÓN RIDGE
# ============================================================================

cat("\n=== ETAPA 6: MODELO LOGÍSTICO (RIDGE REGRESSION) ===\n")

# Calcular pesos de clase para balancear desbalanceo
n_total <- length(y_train)
n_class_0 <- sum(y_train == 0)
n_class_1 <- sum(y_train == 1)

weight_0 <- n_total / (2 * n_class_0)
weight_1 <- n_total / (2 * n_class_1)
pesos <- ifelse(y_train == 0, weight_0, weight_1)

cat("Pesos de clase:\n")
cat("  Clase 0 (No retrasado): ", round(weight_0, 3), "\n", sep = "")
cat("  Clase 1 (Retrasado):    ", round(weight_1, 3), "\n", sep = "")

# Entrenar modelo Ridge (alpha = 0)
set.seed(2026)
modelo_logistico <- glmnet(
  x = X_train,
  y = y_train,
  weights = pesos,
  family = "binomial",
  alpha = 0,  # Ridge regression
  standardize = TRUE
)

cat("✓ Modelo Ridge entrenado\n")
cat("  Valores lambda probados: ", length(modelo_logistico$lambda), "\n", sep = "")

# Validación cruzada para seleccionar lambda óptimo
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
cat("✓ Validación cruzada completada\n")
cat("  Lambda óptimo (CV mínimo): ", round(lambda_optimo, 6), "\n", sep = "")

# ============================================================================
# ETAPA 7: CREAR VARIABLES INTERPRETABLES PARA RPART
# ============================================================================

cat("\n=== ETAPA 7: CREACIÓN DE VARIABLES INTERPRETABLES ===\n")

# Función para categorizar riesgo basado en tasa de retrasos
categorizar_riesgo <- function(data, id_col, target_col) {
  # Calcular tasa de retrasos por categoría
  tasas <- data %>%
    group_by(!!sym(id_col)) %>%
    summarise(
      tasa_retraso = mean(as.numeric(!!sym(target_col)) - 1, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Cuantiles para clasificación
  q33 <- quantile(tasas$tasa_retraso, 0.33, na.rm = TRUE)
  q67 <- quantile(tasas$tasa_retraso, 0.67, na.rm = TRUE)
  
  # Asignar riesgo
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
  # hora debe estar en formato HHMM como character
  if (is.na(hora)) return(NA)
  
  hora_num <- as.numeric(substr(sprintf("%04d", as.numeric(hora)), 1, 2))
  
  case_when(
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

cat("Categorizando variables de riesgo...\n")

# Categorizar aeropuertos de origen
riesgo_origen <- categorizar_riesgo(datos_train, "OriginAirportID", "ArrDel15")
cat("  ✓ Origen: ", paste(table(riesgo_origen$riesgo), collapse = " | "), "\n", sep = "")

# Categorizar aeropuertos de destino
riesgo_destino <- categorizar_riesgo(datos_train, "DestAirportID", "ArrDel15")
cat("  ✓ Destino: ", paste(table(riesgo_destino$riesgo), collapse = " | "), "\n", sep = "")

# Aplicar transformaciones a datos de entrenamiento
datos_train_interpretable <- datos_train %>%
  left_join(riesgo_origen %>% select(OriginAirportID, riesgo), 
            by = "OriginAirportID", suffix = c("", "_origen")) %>%
  rename(OriginRiesgo = riesgo) %>%
  left_join(riesgo_destino %>% select(DestAirportID, riesgo),
            by = "DestAirportID", suffix = c("", "_destino")) %>%
  rename(DestRiesgo = riesgo) %>%
  mutate(
    OriginRiesgo = factor(OriginRiesgo, levels = c("Bajo", "Medio", "Alto")),
    DestRiesgo = factor(DestRiesgo, levels = c("Bajo", "Medio", "Alto")),
    HoraSalida = sapply(CRSDepTime, categorizar_hora),
    HoraLlegada = sapply(CRSArrTime, categorizar_hora),
    Estacion = factor(categorizar_estacion(Month), levels = c("Primavera", "Verano", "Otoño", "Invierno"))
  ) %>%
  select(-OriginAirportID, -DestAirportID, -CRSDepTime, -CRSArrTime, -Month)

# Aplicar las mismas transformaciones a datos de prueba
datos_test_interpretable <- datos_test %>%
  left_join(riesgo_origen %>% select(OriginAirportID, riesgo),
            by = "OriginAirportID", suffix = c("", "_origen")) %>%
  rename(OriginRiesgo = riesgo) %>%
  left_join(riesgo_destino %>% select(DestAirportID, riesgo),
            by = "DestAirportID", suffix = c("", "_destino")) %>%
  rename(DestRiesgo = riesgo) %>%
  mutate(
    OriginRiesgo = factor(OriginRiesgo, levels = c("Bajo", "Medio", "Alto")),
    DestRiesgo = factor(DestRiesgo, levels = c("Bajo", "Medio", "Alto")),
    HoraSalida = sapply(CRSDepTime, categorizar_hora),
    HoraLlegada = sapply(CRSArrTime, categorizar_hora),
    Estacion = factor(categorizar_estacion(Month), levels = c("Primavera", "Verano", "Otoño", "Invierno"))
  ) %>%
  select(-OriginAirportID, -DestAirportID, -CRSDepTime, -CRSArrTime, -Month)

cat("✓ Variables interpretables creadas\n")

# ============================================================================
# ETAPA 8: MODELO DE ÁRBOL DE CLASIFICACIÓN (RPART)
# ============================================================================

library(rpart)
library(rpart.plot)

cat("\n=== ETAPA 8: ÁRBOL DE CLASIFICACIÓN (RPART) ===\n")

# Usar variables interpretables
datos_train_rpart <- datos_train_interpretable
datos_test_rpart <- datos_test_interpretable

# Sincronizar niveles de factor entre train y test
for (col in names(datos_test_rpart)) {
  if (is.factor(datos_test_rpart[[col]])) {
    levels(datos_test_rpart[[col]]) <- levels(datos_train_rpart[[col]])
  }
}

cat("\n--- Distribución de clases en entrenamiento ---\n")
print(table(datos_train_rpart$ArrDel15))

# Calcular pesos de clase para balancear el desbalanceo 80/20
n_train_rpart <- nrow(datos_train_rpart)
n_no <- sum(datos_train_rpart$ArrDel15 == "No")
n_si <- sum(datos_train_rpart$ArrDel15 == "Si")

weight_no <- n_train_rpart / (2 * n_no)
weight_si <- n_train_rpart / (2 * n_si)
pesos_rpart <- ifelse(datos_train_rpart$ArrDel15 == "No", weight_no, weight_si)

cat("\nPesos de clase:\n")
cat("  Clase No: ", round(weight_no, 3), "\n", sep = "")
cat("  Clase Si: ", round(weight_si, 3), "\n", sep = "")

# Entrenar árbol de clasificación
cat("\nEntrenando árbol de clasificación...\n")
set.seed(2026)
arbol_fit <- rpart(
  ArrDel15 ~ .,
  method = "class",
  data = datos_train_rpart,
  weights = pesos_rpart,
  control = rpart.control(cp = 0.001, minsplit = 10, minbucket = 5, xval = 5)
)

cat("✓ Árbol entrenado\n")

# Tabla de poda: mostrar opciones disponibles
cat("\nTabla CP:\n")
print(arbol_fit$cptable)

# Seleccionar el CP con menor xerror (error de validación cruzada)
cp_optimo <- arbol_fit$cptable[which.min(arbol_fit$cptable[, "xerror"]), "CP"]
xerror_min <- min(arbol_fit$cptable[, "xerror"])

cat("\nCP óptimo: ", round(cp_optimo, 6), "\n", sep = "")
cat("Xerror mínimo: ", round(xerror_min, 6), "\n", sep = "")

# Podar el árbol
arbol_pruned <- rpart::prune(arbol_fit, cp = cp_optimo)
cat("✓ Árbol podado\n")

# Visualizar el árbol
cat("\nGenerando visualización del árbol...\n")
rpart.plot(arbol_pruned, main = "Árbol de Clasificación - Retrasos Aéreos (ArrDel15)")

# Predicciones en datos de prueba
cat("\n--- Evaluación en datos de prueba ---\n")
predicciones_arbol <- predict(arbol_pruned, newdata = datos_test_rpart, type = "class")

# Matriz de confusión
confusion_matrix <- table(Predicho = predicciones_arbol, Real = datos_test_rpart$ArrDel15)
cat("Matriz de confusión:\n")
print(confusion_matrix)

# Exactitud
exactitud <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\nExactitud: ", round(exactitud * 100, 2), "%\n", sep = "")

