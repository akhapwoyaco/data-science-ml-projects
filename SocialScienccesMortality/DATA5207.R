## ----include=FALSE---------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = T,warning=FALSE,message=FALSE,attr.source='.numberLines',attr.output='.numberLines', fig.align='center', dpi=350, fig.width=15, fig.height=15)


## --------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(readxl)
library(readxl)
library(janitor)
library(reshape) # cor data reshape
library(broom)


## --------------------------------------------------------------------------------------------------------------------
additional_measure <- read_excel(
  "2018 County Health Rankings Data - v2.xls", 
  sheet = "Additional Measure Data", skip = 1) |>
  clean_names() |>
  select(
    fips, state, county, 
    age_adjusted_mortality, 
      # additional
      # income (median household income, income inequality)
      household_income,
      # population
      population,
      # demography
      percent_african_american, 
      percent_american_indian_alaskan_native, 
      percent_asian, 
      percent_native_hawaiian_other_pacific_islander, 
      percent_hispanic, 
      percent_non_hispanic_white,
      # health care access and quality (
      # percentage of adults aged less than 65 without insurance, 
      percent_uninsured_51,
      # healthcare costs
      costs,
      # number of primary care providers per 100,000 population,
      other_pcp_rate
  )
ranked_measure <- read_excel(
  "2018 County Health Rankings Data - v2.xls", 
  sheet = "Ranked Measure Data", skip = 1)  |>
  clean_names() |> 
  select(
    fips, state, 
    
      # ranked measure
      # income (median household income, income inequality)
      income_ratio,
      # preventable hospitalizations per 1,000 population), 
      preventable_hosp_rate,
      # socioenvironmental factors (
      # average high school freshman graduation rate per 1,000 population, 
      graduation_rate,
      # percentage of adults aged 25 or older with a 4-year college degree, 
      percent_some_college,
      # percentage of single-parent households, 
      percent_single_parent_households,
      # percentage of children living below federal poverty guidelines), and 
      percent_children_in_poverty,
      # behavior factors (percentages of adult obesity and smoking).
      percent_obese,
      percent_smokers
  )
# ranked_measure


## --------------------------------------------------------------------------------------------------------------------
all_data = merge(
  additional_measure,
  ranked_measure
)
# exclude county, state anf fips variables
analysis_data = all_data[, c(-1,-2,-3)]


## --------------------------------------------------------------------------------------------------------------------
summary(analysis_data) |>
  pander::pander()


## --------------------------------------------------------------------------------------------------------------------
correlation_mat <- round(cor(analysis_data, use = 'complete.obs'),1) #
correlation_mat_melt <- melt(correlation_mat)
correlation_mat_melt |> 
  mutate(
    X1 = str_replace(X1, 'percent_', ''), 
    X2 = str_replace(X2, 'percent_', '')) |> 
  ggplot(aes(x = X1, y = X2, fill = value)) +
  geom_tile() +
  geom_text(aes(x = X1, y = X2, label = value)) +
  theme(
    legend.position = 'none',
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


## --------------------------------------------------------------------------------------------------------------------
model_1 = lm(
  age_adjusted_mortality ~ ., data = analysis_data
)
model_1_summary = summary(model_1)


## ----tab.cap='Model Coefficients'------------------------------------------------------------------------------------
tidy(model_1_summary)


## ----tab.cap='Model Summary'-----------------------------------------------------------------------------------------
broom::glance(model_1)


## --------------------------------------------------------------------------------------------------------------------
par(mfrow = c(2, 2))
plot(model_1)
par(mfrow = c(1,1))

