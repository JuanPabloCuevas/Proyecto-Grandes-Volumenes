library(dplyr); library(readr)

data <- read_csv("airline_2m.csv")

colnames(data)

# Eliminamos Diverted Airport Information (Información sobre vuelos desviados)
colnames(data)[startsWith(colnames(data), "Div")]
data <- data %>% 
  select(-starts_with("Div"))

data <- data %>% 
  dplyr::select(
    -any_of(c(
      "Flights", "Flight_Number_Reporting_Airline",
      "OriginAirportSeqID", "OriginCityMarketID", "Origin", 
      "OriginCityName", "OriginStateFips", "OriginStateName", "OriginWac", 
      "DestAirportSeqID", "DestCityMarketID",
      "Dest", "DestCityName", "DestStateFips", "DestStateName", "DestWac"
    ))
  )

library(dplyr)
library(tidyr)
library(ggplot2)

data %>% 
  filter(ArrDel15 == 1, !is.na(CarrierDelay)) %>% 
  select(CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay) %>% 
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>% 
  pivot_longer(cols = everything(), names_to = "TipoDelay", values_to = "Promedio") %>% 
  mutate(TipoDelay = recode(TipoDelay,
                            CarrierDelay = "Aerolínea",
                            WeatherDelay = "Clima",
                            NASDelay = "Sistema aéreo",
                            SecurityDelay = "Seguridad",
                            LateAircraftDelay = "Aeronave previa"
  )) %>% 
  ggplot(aes(x = reorder(TipoDelay, -Promedio), y = Promedio, fill = Promedio)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(
    low = "#F5C6CB",   # rojo claro
    high = "#B71731"   # tu color principal
  ) +
  theme_minimal() +
  labs(
    title = "Promedio de tipos de retraso (vuelos atrasados)",
    x = "Tipo de retraso",
    y = "Minutos promedio"
  ) +
  theme(
    legend.position = "none"
  )

data <- data %>% 
  filter(!is.na(ArrDel15)) 

table(data$ArrDel15)
prop.table(table(data$ArrDel15))

library(ggplot2)

ggplot(data, aes(x = factor(DayOfWeek), fill = factor(ArrDel15))) +
  geom_bar(position = "fill") +
  labs(y = "Proporción", fill = "Atraso")

tabla <- data %>% 
  filter(ArrDel15 == 1) %>% 
  select(ArrDelay, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay) %>%
  filter(!is.na(CarrierDelay)) %>%
  head(15)

tabla

tabla1 <- data %>% 
  filter(ArrDel15 == 1) %>% 
  summarise(
    CarrierDelay = mean(CarrierDelay, na.rm = TRUE),
    WeatherDelay = mean(WeatherDelay, na.rm = TRUE),
    NASDelay = mean(NASDelay, na.rm = TRUE),
    SecurityDelay = mean(SecurityDelay, na.rm = TRUE),
    LateAircraftDelay = mean(LateAircraftDelay, na.rm = TRUE)
  )

tabla1

library(knitr)
library(kableExtra)

kable(tabla1) %>% 
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover"))

library(dplyr)
library(tidyr)
library(ggplot2)

plot1 <- data %>% 
  filter(ArrDel15 == 1, !is.na(CarrierDelay)) %>% 
  select(CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay) %>% 
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>% 
  pivot_longer(cols = everything(), names_to = "TipoDelay", values_to = "Promedio") %>% 
  ggplot(aes(x = TipoDelay, y = Promedio, fill = TipoDelay)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    title = "Promedio de tipos de delay (vuelos atrasados)",
    x = "Tipo de delay",
    y = "Minutos promedio"
  ) +
  theme(legend.position = "none")

plot1

library(dplyr)
library(tidyr)
library(ggplot2)

data %>% 
  filter(ArrDel15 == 1, !is.na(CarrierDelay)) %>% 
  select(CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay) %>% 
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>% 
  pivot_longer(cols = everything(), names_to = "TipoDelay", values_to = "Promedio") %>% 
  mutate(TipoDelay = recode(TipoDelay,
                            CarrierDelay = "Aerolínea",
                            WeatherDelay = "Clima",
                            NASDelay = "Sistema aéreo",
                            SecurityDelay = "Seguridad",
                            LateAircraftDelay = "Aeronave previa"
  )) %>% 
  ggplot(aes(x = TipoDelay, y = Promedio, fill = TipoDelay)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    title = "Promedio de tipos de retraso (vuelos atrasados)",
    x = "Tipo de retraso",
    y = "Minutos promedio"
  ) +
  theme(legend.position = "none")
