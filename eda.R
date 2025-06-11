library(readr)
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train = read_csv('train.csv')

#xgb, multinomial logistic regression, linear svm, 
#knn on each crop separately 

# categorical features
# could possibly be one-hot encoded
train %>% 
  group_by(`Soil Type`) %>% 
  summarize(n=n())

train %>% 
  group_by(`Crop Type`) %>% 
  summarize(n=n())

# outcome
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(n=n())

# temp
# not sure temp is predictive
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Temparature))

# humidity
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Humidity))

# moisture
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Moisture))

# nitrogen
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Nitrogen))

# potassium
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Potassium))

# phos
train %>% 
  group_by(`Fertilizer Name`) %>% 
  summarize(avg_temp=mean(Phosphorous))

