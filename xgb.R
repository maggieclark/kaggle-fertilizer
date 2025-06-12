library(xgboost)
library(dplyr)
library(readr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train_raw = read_csv('train.csv')


### preprocessing ###

# create folds

set.seed(117)

shuffled_indices <- sample(1:750000)

group_assignments <- cut(
  shuffled_indices,
  breaks = c(0, 150000 * 1:5),
  labels = 1:5,
  include.lowest = TRUE, # include the first value in the first interval
  right = TRUE # intervals are (a, b], except for the first if include.lowest = TRUE
)

train = cbind(train_raw, fold = group_assignments)

# code categorical features

dummies <- model.matrix(~ `Soil Type` + `Crop Type` - 1, # -1 to remove intercept
                        data = train)

train = train %>% 
  select(!c(`Soil Type`,`Crop Type`)) %>% 
  cbind(dummies)

# convert text categories to integers

train$`Fertilizer Name` <-
  as.numeric(as.factor(train$`Fertilizer Name`)) - 1


### train ###
# default booster is gbtree (tree-based)

train_x = train %>% 
  filter(fold %in% c(1:4)) %>% 
  dplyr::select(!c(id, fold, `Fertilizer Name`)) %>% 
  as.matrix()

train_y = train %>% 
  filter(fold %in% c(1:4)) %>% 
  dplyr::select(`Fertilizer Name`) %>% 
  as.matrix()

test_x = train %>% 
  filter(fold == 5) %>% 
  dplyr::select(!c(id, fold, `Fertilizer Name`)) %>% 
  as.matrix()

test_y = train %>% 
  filter(fold == 5) %>% 
  dplyr::select(`Fertilizer Name`) %>% 
  as.matrix()

# baseline
# 

# want to use softprob and make multiple predictions
mod <- xgboost(data = train_x,
               label = train_y,
               params = list(num_class=7, 
                             objective = "multi:softmax"),
               nrounds = 10)

yhat = predict(mod, test_x, validate_features = TRUE)

# this is just basic accuracy for now
sum(yhat==test_y)/length(yhat)

#
#
#


# 1000 rounds
# 0.6 ish (maxed out 1000 allowed rounds, default learning rate)

mod <- xgboost(data = train_x, 
               label = train_y, 
               nrounds = 1000, 
               objective = "reg:squaredlogerror",
               eval_set=0.2,
               #learning_rate=1,
               early_stopping_rounds = 1)

summary(mod)
mod$evaluation_log
mod$best_iteration

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

# select best early_stopping_rounds (eval_set required) and learning_rate
# baseline for this will be actually getting it to early-stop
# 0.07 ish 
# (very wonky - only 10 evaluation obs and these overlap with val set, and only one early stop round)

train_mat = xgb.DMatrix(data=train_x,
                     label=train_y)

val_mat = xgb.DMatrix(data=as.matrix(fold5[1:10,c(2:7,9)]),
                      label=as.matrix(fold5[1:10,8]))

evals1 = list(train=train_mat, eval=val_mat)

mod <- xgb.train(data = train_mat,
                 nrounds = 1000,
                 watchlist = evals1,
                 early_stopping_rounds=1,
                 objective="reg:squaredlogerror")

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

# select best early_stopping_rounds (eval_set required) and learning_rate
# test 5 learning rates at each value of early_stopping_rounds

# early stopping = 5

folds = list(fold1, fold2, fold3, fold4, fold5)

learning_rates = c(0.1, 0.3, 0.5, 0.7, 0.9)

es5_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                        label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  

  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=5,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es5_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

# early stopping = 10

es10_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                        label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  

  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=10,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es10_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

# early stopping = 15

es15_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                         label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  
  
  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=15,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es15_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

### train on all data and predict ###

# use learning rate 0.3
# use 15 early stopping rounds
# expected prediction accuracy is around 0.0625

# eval data (subset of training)
eval_index = sample(1:750000, 150000)

eval_dataset = train[eval_index,]

eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                       label=as.matrix(eval_dataset[,8]))

# remaining training data
train_dataset = train[-eval_index,]

train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                        label=as.matrix(train_dataset[,8]))


# test data

test = read_csv('test.csv')

test <- test %>% 
  mutate(Female = case_when(Sex=='female' ~ 1,
                            Sex=='male'~0)) %>% 
  dplyr::select(!Sex)

test_dataset = test %>% 
  dplyr::select(!id) %>% 
  as.matrix()

# mod

w = list(train=train_mat, eval=eval_mat)

mod <- xgb.train(data = train_mat,
                 nrounds = 1000,
                 watchlist = w,
                 early_stopping_rounds=15,
                 objective="reg:squaredlogerror",
                 learning_rate=0.3)

yhat = predict(mod, test_dataset, validate_features = TRUE)

# print

submission = data.frame(id=test$id, Calories=yhat)

write_csv(submission, 'mc_submission_5.29_1.csv')

