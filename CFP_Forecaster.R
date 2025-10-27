## ----document_setup, echo=F, message=F, warning=F--------------------------------------------------------
# This chunk can include things you need for the rest of the document
library('ggplot2') ## most of the time you will need ggplot
theme_set(theme_bw()) # change the default ggplot theme to black-and-white

knitr::opts_chunk$set(
  echo=T, ## show your R code chunk
  message = F, ## hide the message
  warning = F, ## hide the warning
  autodep = T ## make sure your separate code chunks can find the dependencies (from other code chunk)
)


## --------------------------------------------------------------------------------------------------------
## YOUR CODE HERE
library(tidyverse, tidymodels)
library(ggplot2)


## --------------------------------------------------------------------------------------------------------
data <- readr::read_csv("Carbon Emission.csv", col_names = T, show_col_types = F) %>% as_tibble()
data %>% glimpse()


## --------------------------------------------------------------------------------------------------------
visdat::vis_miss(data)


## --------------------------------------------------------------------------------------------------------
visdat::vis_dat(data)


## --------------------------------------------------------------------------------------------------------
# renaming columns for ease of representation
data <- data %>% rename(Body_Type = `Body Type`) %>% rename(Shower_Freq = `How Often Shower`) %>% 
  rename( Heating_Source = `Heating Energy Source`) %>% rename(Vehicle_Type = `Vehicle Type`) %>% 
  rename(Social_Activity = `Social Activity`) %>% rename(Grocery_CostPM = `Monthly Grocery Bill`) %>%
  rename(AirTravel_Freq = `Frequency of Traveling by Air`) %>% rename(Dist_TravelledPM = `Vehicle Monthly Distance Km`) %>%
  rename( Waste_bag_Size = `Waste Bag Size`) %>% rename(Waste_CountPW = `Waste Bag Weekly Count`) %>%
  rename(TV_PC_WatchtimePD = `How Long TV PC Daily Hour`) %>% rename(New_Clothes_PM = `How Many New Clothes Monthly`) %>%
  rename(Internet_TimeH = `How Long Internet Daily Hour`) %>% rename(Energy_Efficiency = `Energy efficiency`)

data %>% glimpse()


## --------------------------------------------------------------------------------------------------------
# factor(data$Body_Type, levels = c("obese", "overweight", "underweight", "normal"))
# factor(data$Diet, levels = c("pescatarian", "vegetarian", "omnivore", "vegan"))
# factor(data$Shower_Freq, levels = c("daily", "less frequently", "more frequently", "twice a day"))
# factor(data$Heating_Source, levels = c("coal", "natural gas", "wood", "electricity"))
# factor(data$Social_Activity, levels = c("often", "never", "sometimes"))
data <- data %>% mutate(Sex = ifelse(Sex == "male", 0, 1))
# 
# data <- data %>%
#   mutate(Waste_bag_Size = recode(Waste_bag_Size, "small" = 1, "medium" = 2, "large" = 3, "extra large" = 4))
# data <- data %>%
#   mutate(AirTravel_Freq = recode(AirTravel_Freq, "never" = 0, "rarely" = 1, "frequently" = 2, "very frequently" = 3))

data <- data %>%
  mutate(Vehicle_Type = ifelse(Transport == "walk/bicycle", "PMM", Vehicle_Type))
data <- data %>%
  mutate(Vehicle_Type = ifelse(Transport == "public", "publicTransport", Vehicle_Type))


## --------------------------------------------------------------------------------------------------------
num_cols <- data %>% select(where(is.numeric))

cat_cols <- data %>% select(where(is.character))


## --------------------------------------------------------------------------------------------------------
num_cols %>% tibble::rowid_to_column() %>% pivot_longer(names(num_cols)) %>% 
  ggplot(mapping = aes(x=value)) + geom_histogram(bins = 20) +
  facet_wrap(~name, scales = 'free')


## --------------------------------------------------------------------------------------------------------
cat_cols %>% select(-c(Recycling, Cooking_With))  %>% pivot_longer(names(.)) %>% 
  ggplot(mapping = aes(x=value)) + geom_bar() +
  facet_wrap(~name, scales = 'free') 


## --------------------------------------------------------------------------------------------------------
options(dlookr_offline = F)
dlookr::plot_normality(num_cols %>% select(CarbonEmission, Grocery_CostPM, Dist_TravelledPM, Waste_CountPW, TV_PC_WatchtimePD, 
                                           New_Clothes_PM, Internet_TimeH ))


## --------------------------------------------------------------------------------------------------------
data %>%
  ggplot(mapping = aes(x= Grocery_CostPM, y= New_Clothes_PM,shape = as.factor(Sex) )) +
  geom_point(aes(color = CarbonEmission), alpha = .75) +
  geom_smooth() +scale_color_viridis_c() +
  # scale_shape_manual(values = c(11,13,14,15,16,17)) +
  facet_wrap(~Body_Type, scales = 'free') + theme_minimal()


## --------------------------------------------------------------------------------------------------------
data %>% mutate(AirTravel_Freq = recode(AirTravel_Freq, "never" = 0, "rarely" = 1, "frequently" = 2, "very frequently" = 3)) %>%
  ggplot(mapping = aes(x= Dist_TravelledPM, y= CarbonEmission , shape = Vehicle_Type)) +
  geom_point(aes(color = AirTravel_Freq), alpha = 0.5) + scale_color_viridis_c() +
  scale_shape_manual(values = c(11,12, 13,14,15,16,17)) +
  facet_wrap(~Transport, scales = 'free') + theme_minimal()


## --------------------------------------------------------------------------------------------------------
data %>% mutate(Energy_Efficiency = recode(Energy_Efficiency, "No" = 0, "Sometimes" = 1, "Yes" = 2)) %>%
  ggplot(mapping = aes(x= TV_PC_WatchtimePD, y= CarbonEmission , shape = Body_Type)) +
  geom_point(aes(color = Energy_Efficiency), alpha = 0.5) + scale_color_viridis_c() +
  facet_wrap(~Social_Activity, scales = 'free') + theme_minimal()


## --------------------------------------------------------------------------------------------------------
data %>% tibble::rowid_to_column() %>% pivot_longer(c("CarbonEmission", "Dist_TravelledPM")) %>%
  ggplot(mapping = aes(x = Vehicle_Type, y= value)) +
  geom_boxplot(aes(fill = as.factor(Sex), color = as.factor(Sex)), alpha = 0.5) +
  facet_wrap(~name, scales = 'free') + theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------------
data %>% ggplot(mapping = aes(x = Heating_Source, y = CarbonEmission)) + geom_boxplot() + 
  facet_wrap(Energy_Efficiency~., scales = 'free_y') + theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------------
data %>% tibble::rowid_to_column() %>% pivot_longer(c("Grocery_CostPM", "Waste_CountPW", "CarbonEmission")) %>%
  ggplot(mapping = aes(x = Waste_bag_Size, y= value)) +
  geom_boxplot() +
  facet_wrap(~name, scales = 'free') + theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------------
check_str_method <- function(type_list, check_val) {
  if (check_val %in% unlist(strsplit(gsub("\\[|\\]|'", "", type_list), ", "))) {
    return(1)
  } else {
    return(0)
  }
}

# Create new columns based on cooking methods
data <- data %>%
  mutate(
    Cooking_Grill = sapply(Cooking_With, check_str_method, check_val = "Grill"),
    Cooking_Oven = sapply(Cooking_With, check_str_method, check_val = "Oven"),
    Cooking_Stove = sapply(Cooking_With, check_str_method, check_val = "Stove"),
    Cooking_Microwave = sapply(Cooking_With, check_str_method, check_val = "Microwave"),
    Cooking_Airfryer = sapply(Cooking_With, check_str_method, check_val = "Airfryer")) %>% select(-Cooking_With)


data <- data %>%
  mutate(
    Recycling_Paper = sapply(Recycling, check_str_method, check_val = "Paper"),
    Recycling_Plastic = sapply(Recycling, check_str_method, check_val = "Plastic"),
    Recycling_Glass = sapply(Recycling, check_str_method, check_val = "Glass"),
    Recycling_Metal = sapply(Recycling, check_str_method, check_val = "Metal")) %>% select(-Recycling)


## --------------------------------------------------------------------------------------------------------
data %>% tibble::rowid_to_column() %>% pivot_longer(c(starts_with('Cooking'), starts_with('Recycling'))) %>%
  ggplot(mapping = aes(x = CarbonEmission, y = value )) + 
  geom_point(aes(color = value), alpha = 0.75) + geom_smooth(method = 'glm', color = 'black') +
  facet_wrap(~name, scales = 'free')


## --------------------------------------------------------------------------------------------------------
data %>% tibble::rowid_to_column() %>% pivot_longer(c(starts_with('Cooking'))) %>%
  ggplot(mapping = aes(x = Diet, y = CarbonEmission)) +
  geom_boxplot(aes(fill = as.factor(value), color = as.factor(value), alpha=0.5 ))  +  facet_wrap(name~., scales = 'free_y') +
  theme_minimal() +   theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------------
dummy_vars <- caret::dummyVars(~., data = data, fullRank = T)
data_oh <- predict(dummy_vars, data) %>%as.data.frame()
data_oh %>% glimpse()


## --------------------------------------------------------------------------------------------------------
abbreviated_column_names <- c("BT_obese", "BT_overweight", "BT_underweight", "S", "D_pescatarian", "D_vegan", "D_vegetarian",
    "SF_less_frequent", "SF_more_frequent", "SF_twice_a_day", "HS_electric", "HS_gas", "HS_wood", "T_public", "T_walk_bike",
    "VT_electric", "VT_hybrid", "VT_lpg", "VT_petrol", "VT_PMM", "VT_public", "SA_often", "SA_sometimes", "GC_PM",
    "ATF_never", "ATF_rarely", "ATF_very_frequent", "DT_PM", "WB_large", "WB_medium", "WB_small", "WCP",
    "TPW", "NCP", "IT", "EE_sometimes", "EE_yes", "CG_grill", "CG_oven", "CG_stove", "CG_microwave", "CG_airfryer",
    "RP_paper", "RP_plastic", "RP_glass", "RP_metal")
df <- data_oh %>% select(-CarbonEmission)
colnames(df) <-  abbreviated_column_names
df %>% cor() %>% corrplot::corrplot(type = 'upper', method = 'square')


## --------------------------------------------------------------------------------------------------------
library(FactoMineR, factoextra)


## --------------------------------------------------------------------------------------------------------
mfa_res <- FactoMineR::FAMD(data %>% select(-CarbonEmission), ncp = 26, graph = F)


## --------------------------------------------------------------------------------------------------------
eig_val <- data.frame(mfa_res$eig) %>% tibble::rowid_to_column() %>% rename(Component = rowid)
eig_val %>% arrange(percentage.of.variance) %>%
  ggplot(mapping = aes(x= Component, y = percentage.of.variance)) + 
    geom_text(aes(label = paste0(round(percentage.of.variance, 2), "%")), 
            vjust = -0.5, size = 3, color = "black") + 
  geom_bar(stat = 'identity', fill = 'blue') +
  geom_line(aes(y = percentage.of.variance), color = 'red', size = 1) +
  labs(x = 'Principal Component', y = 'Variance Explained(%)', title = "Scree Plot for FAMD Eigen Values") +
  theme_minimal() + 
     theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_famd_var(mfa_res, repel = T)


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_famd_var(mfa_res, "quanti.var", col.var = "contrib", 
                          gradient.cols = c("red", "green", "blue"),repel = T)


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_contrib(mfa_res, 'var', axes = 2)


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_contrib(mfa_res, 'var', axes = 3)


## --------------------------------------------------------------------------------------------------------
sim <- mfa_res$quali.var$cos2[,1]
mds_result <- data.frame(mfa_res$quali.var$contrib[,1:2])
mds_result %>% ggplot() +
  ggrepel::geom_text_repel(aes(label = rownames(mds_result), x = Dim.1, y = Dim.2 , color =sim))


## --------------------------------------------------------------------------------------------------------
# reordering the column names to order them in groups
data_mfa <- data %>% select("Body_Type", "Diet", "Shower_Freq", "Heating_Source", "Energy_Efficiency", "Transport", "AirTravel_Freq", "Vehicle_Type",  "Waste_bag_Size", "Social_Activity", "Sex", "Grocery_CostPM", "Cooking_Grill", "Cooking_Oven", "Cooking_Stove", "Cooking_Microwave", "Cooking_Airfryer" , "Dist_TravelledPM", "Waste_CountPW", "Recycling_Paper", "Recycling_Plastic", "Recycling_Glass", "Recycling_Metal", "TV_PC_WatchtimePD", "New_Clothes_PM", "Internet_TimeH")

data_scaled <- data_mfa %>% select(Grocery_CostPM, Dist_TravelledPM, TV_PC_WatchtimePD, New_Clothes_PM, Internet_TimeH) %>% 
  scale(center = T, scale = T) %>% as_tibble()
 
data_multi <- data_mfa %>% select(-c(names(data_scaled))) %>% bind_cols(data_scaled)
  
data_multi %>% glimpse()


## --------------------------------------------------------------------------------------------------------
multi_res <- MFA(data_mfa, group = c(3,2,3,2,1,5,5,5), type = c(rep('n',4),rep('c',3), 's'), ncp = 26, 
    name.group = c('Body','heat', 'trans','waste','Sex', 'cook','waste','groc'), graph = F)


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_mfa_var(multi_res, "quali.var", col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             col.var.sup = "violet", repel = TRUE,
             geom = c("point", "text"))



## --------------------------------------------------------------------------------------------------------
factoextra::fviz_mfa_var(multi_res, 'quali.var', col.var = "cos2", repel = T)


## --------------------------------------------------------------------------------------------------------
multi_dims <- data.frame(multi_res$group$coord[,1:2])
multi_dims %>% ggplot() +
  ggrepel::geom_text_repel(aes(label = rownames(multi_dims), x = Dim.1, y = Dim.2 )) #


## --------------------------------------------------------------------------------------------------------
corrplot::corrplot(multi_res$group$correlation)


## --------------------------------------------------------------------------------------------------------
factoextra::fviz_contrib(multi_res, "quanti.var", repel = T)



## --------------------------------------------------------------------------------------------------------
factoextra::fviz_contrib(multi_res, "quali.var", repel = T)


## ----train_test_split------------------------------------------------------------------------------------
set.seed(1234)
library(rsample)
data_split <- initial_split(data, strata = 'Vehicle_Type', prop = 0.85)
train_data <- training(data_split)
test_data <- testing(data_split)


## ----scale_transformation--------------------------------------------------------------------------------
col_names <- c("Grocery_CostPM", "Dist_TravelledPM", "TV_PC_WatchtimePD", "New_Clothes_PM", "Internet_TimeH" )
var_mean <- as.vector(apply(train_data[,col_names], 2, mean))
var_sd <- as.vector(apply(train_data[,col_names], 2, sd))
df_train  <- train_data[,col_names] %>% scale(center = var_mean, scale = var_sd) %>% 
  as.data.frame() %>% tibble::as_tibble() %>%
  bind_cols(train_data %>% select(-col_names))

df_train %>% glimpse()

var_mean
var_sd


## ----test_data-------------------------------------------------------------------------------------------
df_test <- test_data[,col_names] %>% scale(center = var_mean, scale = var_sd) %>%
  as.data.frame() %>% tibble::as_tibble() %>%
  bind_cols(test_data%>% select(-col_names))

df_test %>% glimpse()

# write.csv(df_test,"testing_data_oh.csv")


## ----linear_models---------------------------------------------------------------------------------------
set.seed(1234)
# numerical variables only 
numerical_only <- df_train %>% select(where(is.numeric))
lm_numeric <- lm(CarbonEmission~., data = numerical_only)

# Categorical variables only
categorical_only <- df_train %>% select(where(is.character), CarbonEmission)
lm_categorical <- lm(CarbonEmission~., data = categorical_only)

# All categorical and numerical columns
lm_all <- lm(CarbonEmission~., data = df_train)

# All categorical main effects and pairwise interactions of numerical variables
lm_lin_cat_pairwise <- lm(CarbonEmission~ (.)^2 , data = df_train)

#Interaction between vehicletype, Transport, Distance travelled with the rest of the features
lm_Vehic_trans_dist <- lm(CarbonEmission~Vehicle_Type*(.)+Transport *(.)+Dist_TravelledPM *(.), data = df_train)

# Non-linear affects of the above variables combined
lm_nonlin_vtd <- lm(CarbonEmission~(Vehicle_Type+Transport+Dist_TravelledPM)^2*(.), data = df_train)


## ----model_summary---------------------------------------------------------------------------------------
model_summary <- tibble()
for (i in c('lm_numeric', 'lm_categorical', 'lm_all', 'lm_lin_cat_pairwise', 'lm_Vehic_trans_dist', 'lm_nonlin_vtd')){
  result <- broom::glance(get(i))
  rmse <- modelr::rmse(get(i), df_test)
  model_summary <- bind_rows(model_summary, tibble(Model = i, rmse = rmse, result))
}


## ----lm_metrics------------------------------------------------------------------------------------------
model_summary %>% select(Model, df, r.squared, AIC, BIC) %>% 
  pivot_longer(!c("Model", "df")) %>% 
  ggplot(mapping = aes(x=Model, y=value, group=value)) +
  labs(title = "Model Comparison", x = "Model", y = "Value") +
  geom_point()+
  facet_wrap(~name, scales = "free_y") +  
  theme(axis.text.x = element_text(angle = 45))


## ----rmse_plot-------------------------------------------------------------------------------------------
model_summary %>% ggplot(mapping = aes(Model, rmse, group=1))+
  geom_line()+geom_point() + theme_minimal()+  
  geom_point(data = model_summary[which.min(model_summary$rmse), , drop = FALSE],color = "red", size = 3) +
  theme(axis.text.x = element_text(angle = 45))


## --------------------------------------------------------------------------------------------------------
model_1 <- lm_nonlin_vtd %>% summary()

tidy_coeffs <- broom::tidy(model_1) %>%
  mutate(upper = estimate + `std.error`, lower = estimate - `std.error`) %>% filter(p.value < 0.001)

tidy_coeffs %>% GGally::ggcoef()




## --------------------------------------------------------------------------------------------------------
lm_lin_cat_pairwise %>% summary() %>% broom::tidy() %>%
  mutate(upper = estimate + `std.error`, lower = estimate - `std.error`) %>% 
  filter(p.value < 0.001) %>% 
  GGally::ggcoef()




## ----tune_params-----------------------------------------------------------------------------------------
library(caret)
set.seed(12345)
my_ctrl<- trainControl(method = 'repeatedcv', number = 10, repeats = 5, savePredictions = T)

my_metric <- 'RMSE'


## ----model1, eval=FALSE----------------------------------------------------------------------------------
# set.seed(1256)
# 
# mod_1 <- train(formula(lm_nonlin_vtd), method='lm', trControl = my_ctrl,
#                metric= my_metric,
#                data = df_train)
# mod_1 %>% readr::write_rds("model_1.rds")


## ----model1_reload---------------------------------------------------------------------------------------
mod_1 <- readr::read_rds("model_1.rds")
plot(varImp(mod_1), top = 30)


## ----model2, eval=FALSE----------------------------------------------------------------------------------
# set.seed(1256)
# 
# mod_2 <- train(formula(lm_lin_cat_pairwise), method='lm', trControl = my_ctrl,
#                metric= my_metric,
#                data = df_train)
# mod_2 %>% readr::write_rds("model_2.rds")


## ----model2_reload---------------------------------------------------------------------------------------
mod_2 <- readr::read_rds("model_2.rds")
plot(varImp(mod_2), top = 30)


## ----model3, eval=F--------------------------------------------------------------------------------------
# set.seed(1256)
# 
# glm_1_def <- train(formula(lm_nonlin_vtd), method='glmnet', trControl = my_ctrl,
#                metric= my_metric,
#                data = df_train)
# lambda_grid <- exp(seq(log(min(glm_1_def$results$lambda)),
#                           log(max(glm_1_def$results$lambda)), length.out = 25))
# enet_grid <- expand.grid(alpha = seq(0.1,1, by=0.1),
#                          lambda = lambda_grid)
# 
# glm_1 <- train(formula(lm_nonlin_vtd), method='glmnet', trControl = my_ctrl,
#                metric= my_metric, tuneGrid = enet_grid,
#                data = df_train)
# glm_1 %>% readr::write_rds("glm_1.rds")


## ----model3_reload---------------------------------------------------------------------------------------
mod_3 <- readr::read_rds("glm_1.rds")
plot(varImp(mod_3),top = 30)


## ----model4, eval=FALSE----------------------------------------------------------------------------------
# set.seed(1256)
# 
# glm_2_def <- train(formula(lm_lin_cat_pairwise), method='glmnet', trControl = my_ctrl,
#                metric= my_metric,
#                data = df_train)
# 
# lambda_grid <- exp(seq(log(min(glm_2_def$results$lambda)),
#                           log(max(glm_2_def$results$lambda)), length.out = 25))
# enet_grid_2 <- expand.grid(alpha = seq(0.1,1, by=0.1),
#                          lambda = lambda_grid)
# 
# glm_2 <- train(formula(lm_lin_cat_pairwise), method='glmnet', trControl = my_ctrl,
#                metric= my_metric, tuneGrid = enet_grid_2,
#                data = df_train)
# # glm_2 %>% readr::write_rds("glm_2.rds")


## ----model4_reload---------------------------------------------------------------------------------------
mod_4 <- readr::read_rds("glm_2.rds")
plot(varImp(mod_4), top = 35)


## ----model5, eval=FALSE----------------------------------------------------------------------------------
# set.seed(12345)
# 
# 
# # my_ctrl_1<- trainControl(method = 'repeatedcv', number = 10, repeats = 5, savePredictions = T, verboseIter = T)
# 
# rf_grid <- expand.grid(mtry = c(1:7),
#                           KEEP.OUT.ATTRS = FALSE,
#                         stringsAsFactors = FALSE)
# 
# rf_tune <- caret::train(CarbonEmission ~ ., method = 'rf',
#                       data = df_train,
#                       importance = TRUE,
#                       metric = my_metric,
#                       trControl = my_ctrl,
#                       tuneGrid = rf_grid)
# rf_tune %>% readr::write_rds("rf_tune.rds")


## ----model5_reload---------------------------------------------------------------------------------------
mod_5 <- readr::read_rds("rf_tune.rds")
plot(varImp(mod_5), top=35)


## ----model6, eval=FALSE----------------------------------------------------------------------------------
# set.seed(12233)
# 
# df_train_1 = readr::read_csv('train_data_oh.csv', show_col_types = F)
# xgb_default <- train( CarbonEmission ~ .,
#                       data = df_train_1,
#                       method = 'xgbTree',
#                       metric = my_metric,
#                       trControl = my_ctrl,
#                       verbosity = 0,
#                       nthread = 1)
# 
# xgb_grid <- expand.grid(nrounds = c(25, 50, 100, 200, 400, 800,1000),
#                         max_depth = c(3, 6, 9, 12),
#                         eta =  c(0.0625*xgb_default$bestTune$eta,
#                                  0.125*xgb_default$bestTune$eta,
#                                  1.0*xgb_default$bestTune$eta),
#                         gamma = xgb_default$bestTune$gamma,
#                         colsample_bytree = xgb_default$bestTune$colsample_bytree,
#                         min_child_weight = xgb_default$bestTune$min_child_weight,
#                         subsample = xgb_default$bestTune$subsample)


## ----model6_tune, eval=F---------------------------------------------------------------------------------
# xgb_tune <-  train( CarbonEmission ~ .,
#                    data = df_train_1,
#                    method = 'xgbTree',
#                    metric = my_metric,
#                    tuneGrid = xgb_grid,
#                    trControl = my_ctrl,
#                    verbosity = 0,
#                    nthread = 4)
# xgb_tune %>% readr::write_rds("xgb_tune_oh.rds")


## ----model6_reload---------------------------------------------------------------------------------------
mod_6 <- readr::read_rds("xgb_tune.rds")
plot(mod_6, xTrans=log)


## ----model6_varimp---------------------------------------------------------------------------------------

plot(varImp(mod_6), top=35)


## ----model7,eval=FALSE-----------------------------------------------------------------------------------
# set.seed(1234543)
# nnet_grid <- expand.grid(size=c(5,9,13,17,21), decay=exp(seq(-6,0,length.out=11)))
# nnet_tune <- train(CarbonEmission~., method='nnet', metric = my_metric, trControl = my_ctrl,
#                trace=F, tuneGrid = nnet_grid, preProcess=c('center', 'scale'), data = df_train)
# nnet_tune %>% readr::write_rds("nnet_tune.rds")


## ----model7_reload---------------------------------------------------------------------------------------
mod_7 <- readr::read_rds("nnet_tune.rds")
plot(mod_7, xTrans=log)


## --------------------------------------------------------------------------------------------------------
calculate_metrics <- function(model, df_test) {
  predictions <- predict(model, newdata = df_test %>% select(-CarbonEmission))
  
  mse <- mean((df_test$CarbonEmission - predictions)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(df_test$CarbonEmission - predictions))
  
  return(data.frame(MSE = mse, RMSE = rmse, MAE = mae))
}

model_list <- list(lin_mod_poly = mod_1, lin_mod_pairwise = mod_2,
                   glmnet_poly = mod_3, glmnet_pairwise = mod_4,
                   rf_tuned = mod_5, xgb_tune = mod_6,
                   nnet_tune = mod_7)

results <- lapply(model_list, calculate_metrics, df_test = df_test)

results_df <- do.call(rbind, results)

results_df <- results_df %>% rownames_to_column('Model')



## --------------------------------------------------------------------------------------------------------
results_df %>%
  tibble::rowid_to_column() %>%
  pivot_longer(c( 'MAE', 'RMSE')) %>%
  ggplot(aes(x = Model, y = value, color = name, group = name)) +
  geom_point() +
  geom_line(size = 1.2) +
  labs(title = "Model Evaluation Metrics",
       x = "Model",
       y = "Metric Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45))+
  facet_wrap(~name, scales = 'free_y')



## --------------------------------------------------------------------------------------------------------
best_preds <- predict(mod_2, df_test %>% select(-CarbonEmission))

df_test %>% select(Vehicle_Type, Transport,AirTravel_Freq,Heating_Source, CarbonEmission) %>% mutate('Preds'=best_preds) %>% group_by(Vehicle_Type, Transport, AirTravel_Freq, Heating_Source) %>% summarise(rmse = caret::RMSE(Preds, CarbonEmission),.groups = 'drop') %>%
  arrange(rmse) %>%  
  ggplot(mapping = aes(x=interaction(Vehicle_Type, Transport, sep = '_'), y=rmse, fill=AirTravel_Freq)) + 
  geom_col(position = 'dodge')+
  ggtitle("Hardest to predict variables for the best model")+
  facet_wrap(~Heating_Source, scales = 'free_y')+
  theme(axis.text.x = element_text(angle = 45))
  


## --------------------------------------------------------------------------------------------------------

df_test %>% select(Dist_TravelledPM, Waste_CountPW,Waste_bag_Size,Body_Type, CarbonEmission) %>% mutate('Preds'=best_preds) %>% group_by(Dist_TravelledPM, Waste_CountPW,Waste_bag_Size,Body_Type) %>% summarise(rmse = caret::RMSE(Preds, CarbonEmission),.groups = 'drop') %>%
  arrange(rmse) %>%  
  ggplot(mapping = aes(x=interaction(Waste_bag_Size, Body_Type, sep = '_'), y=rmse, fill=Waste_CountPW)) + 
  geom_col(position = 'dodge')+
  ggtitle("Hardest to predict variables for the best model")+
  # facet_wrap(~Heating_Source, scales = 'free_y')+
  theme(axis.text.x = element_text(angle = 45))
  



## --------------------------------------------------------------------------------------------------------
residaul_plot <- data.frame(actual = df_test$CarbonEmission, predictions = best_preds)

residaul_plot$residuals <- df_test$CarbonEmission-best_preds
residaul_plot$average <- with(residaul_plot, (actual + predictions) / 2)
residaul_plot$difference <- with(residaul_plot, actual - predictions)
mean_diff <- mean(residaul_plot$difference)
sd_diff <- sd(residaul_plot$difference)

# Bland-Altman Plot
ggplot(residaul_plot, aes(x = average, y = difference)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = mean_diff, color = "blue") +
  geom_hline(yintercept = mean_diff + 1.96 * sd_diff, linetype = "dashed", color = "red") +
  geom_hline(yintercept = mean_diff - 1.96 * sd_diff, linetype = "dashed", color = "red") +
  labs(x = "Average of actual and predicted", y = "Difference (Actual - Predicted)",
       title = "Bland-Altman (Residual vs Predicted) Plot")



## --------------------------------------------------------------------------------------------------------
library(pdp)

interaction_pdp <- partial(
  mod_2,
  pred.var = c("Vehicle_Type", "Transport"),
  train = df_train,
  grid.resolution = 10  )

interaction_pdp %>%
  ggplot() +
  geom_tile(aes(x = Vehicle_Type, y = Transport, fill = yhat)) +  # Map x, y, and fill
  geom_contour(aes(x = Vehicle_Type, y = Transport, z = yhat), colour = "white", bins = 10) +  # Add x, y, and z
  scale_fill_viridis_c(option = "C") +  # Use a color gradient from the viridis package
  labs(
    title = "Partial Dependence Plot for Interaction",
    x = "Vehicle Type",  # Adjusted for your specific feature names
    y = "Transport",
    fill = "Predicted Outcome"  )+
  theme(axis.text.x = element_text(angle = 45))




## --------------------------------------------------------------------------------------------------------
library(pdp)

interaction_2 <- partial(
  mod_2,
  pred.var = c("Dist_TravelledPM", "Waste_CountPW"),
  train = df_train,
  grid.resolution = 10  )

interaction_2 %>%
  ggplot() +
  geom_tile(aes(x = Dist_TravelledPM, y = Waste_CountPW, fill = yhat)) +  # Map x, y, and fill
  geom_contour(aes(x = Dist_TravelledPM, y = Waste_CountPW, z = yhat), colour = "white", bins = 10) +  # Add x, y, and z
  scale_fill_viridis_c(option = "C") +  # Use a color gradient from the viridis package
  labs(
    title = "Partial Dependence Plot for Interaction",
    x = "Distance Travelled Per month",  # Adjusted for your specific feature names
    y = "Waste bag count",
    fill = "Predicted Outcome"  )+
  theme(axis.text.x = element_text(angle = 45))



