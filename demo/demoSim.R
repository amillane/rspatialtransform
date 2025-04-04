### Demo ###

#We will be using tidymodels for this demo so we need to install the package
#Any other machine learning package in R will work as well
# devtools::install_github("amillane/rspatialtransform")
library(tidymodels)
library(vip)
library(rspatialtransform)
library(doParallel)

#Load in the data
#load("SimulatedData1.RData")
load("NonLinSimDataSet1.RData")




#Plotting the data
ggplot(data=trainData, aes(x=trainLocs[,1], y=trainLocs[,2], color=y)) +
  geom_point() +
  scale_color_distiller(palette="Spectral") +
  theme_minimal() +
  ggtitle("Training Data")



cores <- parallel::detectCores()-5

### Tranform the data to independent data
time_result <- system.time(
  transformedData <- transform_to_ind(formula=y~.,
                                      trainData=trainData,
                                      trainLocs=trainLocs,
                                      testData=testData[,-1],
                                      testLocs=testLocs,
                                      smoothness=1/2,
                                      M=30,
                                      ncores=cores)
)
print(time_result)



## Creating a recipe for both a spatial and non-spatial random forest

#Spatial Recipe
spatial_rf_rec <- recipe(y~., data=transformedData$trainData)

#Non-Spatial Recipe
rf_rec <- recipe(y~., data=trainData)

#Initialize the random forest model
rf_model <- rand_forest(mtry=tune(), min_n = tune(), trees=100) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

#Spatial Workflow
spatial_rf_wf <- workflow() %>%
  add_recipe(spatial_rf_rec) %>%
  add_model(rf_model)

#None-Spatial Workflow
rf_wf <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_model)


## RF tuning grid
rf_grid <- grid_regular(mtry(range=c(1,4)),
                        min_n(),
                        levels=5)

## Setup CV folds
spatial_folds <- vfold_cv(data=transformedData$trainData, v=5)
no_spatial_folds <- vfold_cv(data=trainData, v=5)

cl <- parallel::makePSOCKcluster(cores)
registerDoParallel(cl)

## Run CV using grid
spatial_rf_cv <- spatial_rf_wf %>%
  tune_grid(resamples=spatial_folds,
            grid=rf_grid,
            metrics=metric_set(rmse))


rf_cv <- rf_wf %>%
  tune_grid(resamples=no_spatial_folds,
            grid=rf_grid,
            metrics=metric_set(rmse))


parallel::stopCluster(cl)


## Finalize According to best tune
spatial_rf_wf <- spatial_rf_wf %>%
  finalize_workflow(select_best(spatial_rf_cv, metric="rmse")) %>%
  fit(data=transformedData$trainData)

rf_wf <- rf_wf %>%
  finalize_workflow(select_best(rf_cv, metric="rmse")) %>%
  fit(data=trainData)


## Make Predictions
spatial_RF_preds <- predict(spatial_rf_wf, new_data=transformedData$testData)$.pred %>%
  back_transform_to_spatial(preds=., transformedData) #back transform to spatial

RF_preds <- predict(rf_wf, new_data=testData)$.pred




## VIP
p1 <- vip(spatial_rf_wf$fit$fit, num_features = 10)
p2 <- vip(rf_wf$fit$fit, num_features = 10)

#The Variable Importance plots do not change much between the spatial and non-spatial random forest

ggpubr::ggarrange(p1, p2, ncol=2)




#Predictions
testPreds <- data.frame(truth=testData$y,
                        SpatialRF=spatial_RF_preds,
                        NonSpatialRF= RF_preds)

results <- data.frame(t(apply(testPreds,2,FUN = function(o){
  sqrt(mean((testPreds$truth-o)^2))
})))



#Comparing the predictions to the truth with graph

t1 <- ggplot(data=testData, aes(x=testLocs[,1], y=testLocs[,2], color=y)) +
  geom_point() +
  scale_color_distiller(palette="Spectral") +
  theme_minimal() +
  ggtitle("Test Data")

t2 <- ggplot(data=testPreds, aes(x=testLocs[,1], y=testLocs[,2],
                                 color=SpatialRF)) +
  geom_point() +
  scale_color_distiller(palette="Spectral") +
  theme_minimal() +
  ggtitle("Spatial Random Forest Predictions")

t3 <- ggplot(data=testPreds, aes(x=testLocs[,1], y=testLocs[,2],
                                 color=NonSpatialRF)) +
  geom_point() +
  scale_color_distiller(palette="Spectral") +
  theme_minimal() +
  ggtitle("Non-Spatial Random Forest Predictions")

ggpubr::ggarrange(t1, t2, t3, ncol=2)
