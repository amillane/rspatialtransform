##
## Run various ML models with spatial and non-spatial ML
##
## Libraries
library(tidyverse)
library(tidymodels)
library(parallel)
library(ranger)
library(nnet)
library(rspatialtransform)

## Load in the Data
load("PM25.RData")
fullTrainData <- PM25 %>%
  filter(!is.na(PM25)) %>%
  select(-ID) #%>%
#  mutate(PM25=log(PM25+.5))
testData <- PM25 %>%
  filter(is.na(PM25)) %>%
  select(-ID)

## Split Domain into squares
nbreaks <- 5
fullTrainData <- fullTrainData %>% 
  mutate(LongInd=as.numeric(cut(X, breaks=nbreaks, include.lowest=TRUE, labels=1:5)),
         LatInd=as.numeric(cut(Y, breaks=nbreaks, include.lowest=TRUE, labels=1:5)),
         cvInd=nbreaks*(LongInd-1)+LatInd)
# ggplot() +
#   geom_polygon(data=usa, mapping=aes(x=long, y=lat, group=group), fill="white") +
#   geom_point(data=fullTrainData, aes(x=X, y=Y, color=factor(cvInd)), size=.5)

## Define train/validation combinations
combos <- expand.grid(lat=1:nbreaks, lon=1:nbreaks)
combos$ind <- 1:nrow(combos)
possible <- combn(sort(unique(fullTrainData$cvInd)), nbreaks)
kp <- c()
for(i in 1:ncol(possible)){
  rcind <- combos[combos$ind%in%possible[,i],]
  if(max(table(rcind$lat))==1 & max(table(rcind$lon))==1){
    kp <- c(kp, i)
  }
}
cvFolds <- possible[,kp]

## Create matrices to hold RMSE across folds and models
RMSE_mat_spatial <- RMSE_mat <- matrix(NA, ncol=ncol(cvFolds), nrow=7)

pb <- txtProgressBar(min=0, max=ncol(cvFolds), style=3)
for(cv in 1:ncol(cvFolds)){
  
  ## Split into test and training
  validData <- fullTrainData %>%
    filter(cvInd%in%cvFolds[,cv]) %>%
    select(-LongInd, -LatInd, -cvInd)
  trainData <- fullTrainData %>%
    filter(!(cvInd%in%cvFolds[,cv])) %>%
    select(-LongInd, -LatInd, -cvInd)
  # ggplot() +
  #   geom_polygon(data=usa, mapping=aes(x=long, y=lat, group=group), fill="white") +
  #   geom_point(data=trainData, aes(x=X, y=Y, color=PM25), size=.5) +
  #   geom_point(data=validData, aes(x=X, y=Y, color=PM25), shape=17) +
  #   scale_color_viridis()
  
  ## CV Folds for Tuning
  nfolds <- 5
  folds <- vfold_cv(trainData, v=nfolds)
  
  ## Set train and valid locs
  trainLocs <- trainData %>%
    select(X,Y) %>%
    as.matrix()
  validLocs <- validData %>%
    select(X,Y) %>%
    as.matrix()
  
  ## Determine a grid of spatial parameters for tuning
  nsGrid <- 10
  D <- fields::rdist(trainData[,c("X","Y")])
  max.dist <- max(D)
  min.dist <- max(apply(D,1,function(x){sort(x)[2]}))
  rng_lb <- fields::Matern.cor.to.range(min.dist,nu=1/2,cor.target=0.05)
  rng_ub <- fields::Matern.cor.to.range(max.dist,nu=1/2,cor.target=0.5)
  sr.seq <- exp(seq(log(rng_lb),log(rng_ub),length=nsGrid))
  pct.spatial <- seq(0,.99,length=nsGrid)
  
  #####################################
  ## Fit & Tune Spatial Linear Model ##
  #####################################
  
  linModel <- fit_NN_Matern(PM25~., 
                            locs=trainLocs, nu=1/2, 
                            NearNeighs=self_mkNNIndx(trainLocs, 30),
                            data=trainData)
  spatLinPreds <- predict_NN_Matern(linModel, predlocs=validLocs, 
                                    newdata=validData)$pred
  RMSE_mat_spatial[1,cv] <- rmse_vec(validData %>% pull(PM25),
                                     spatLinPreds)
  
  
  linmod <- lm(PM25~., data=trainData)
  linPreds <- predict(linmod, newdata=validData)
  RMSE_mat[1,cv] <- rmse_vec(validData %>% pull(PM25),
                             linPreds)
  
  ############################
  ## Fit Rand Forest Models ##
  ############################
  
  ## Determine Grid of Tuning Parameters
  tuningGrid <- expand.grid(range=sr.seq,
                            nugget=pct.spatial,
                            fold=1:nfolds,
                            mtry=1:8,
                            min.node.size=seq(5,25,by=5))
  
  ## Function to get RF RMSE for tuning
  get_RF_RMSE <- function(rng, nug, fld=NA, Mt, mns){
    ## Transform to independent
    trndata <- get_rsplit(folds, index=fld) %>%
      analysis(.)
    trnlocs <- get_rsplit(folds, index=fld) %>%
      analysis(.) %>%
      select(X, Y) %>%
      as.matrix()
    vlddata <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(-PM25)
    vldlocs <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(X, Y) %>%
      as.matrix()
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores=5)
    
    ## Fit Model
    rf_mod <- ranger(y~.,
                     data=indData$trainData,
                     num.trees=500,
                     mtry=Mt,
                     min.node.size=mns)
    
    ## Get Predictions
    rf_preds <- predict(rf_mod, data=indData$testData)$predictions
    
    ## Back Transform
    rf_preds <- back_transform_to_spatial(preds=rf_preds, indData)
    
    ## Retain Assessment
    rmseVal <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      pull(PM25) %>%
      rmse_vec(., rf_preds)
    return(rmseVal)
  }
  
  ## Function to get RF preds
  get_RF_preds <- function(rng, nug, Mt, mns){
    ## Transform to independent
    trndata <- trainData
    trnlocs <- trainLocs
    vlddata <- validData %>%
      select(-PM25)
    vldlocs <- validLocs
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    rf_mod <- ranger(y~.,
                     data=indData$trainData,
                     num.trees=500,
                     mtry=Mt,
                     min.node.size=mns)
    
    ## Get Predictions
    rf_preds <- predict(rf_mod, data=indData$testData)$predictions
    
    ## Back Transform
    rf_preds <- back_transform_to_spatial(preds=rf_preds, indData)
    
    ## Retain Preds
    return(rf_preds)
  }
  
  ## Run the Function for all tuning parameters
  tuningResults <- parallel::mclapply(1:nrow(tuningGrid), FUN=function(ind){
    return(get_RF_RMSE(rng=tuningGrid$range[ind],
                       nug=tuningGrid$nugget[ind],
                       fld=tuningGrid$fold[ind],
                       Mt=tuningGrid$mtry[ind],
                       mns=tuningGrid$min.node.size[ind]))
  }, mc.cores=parallel::detectCores()-5) %>%
    do.call("c", args=.)
  
  bestTune <- tuningGrid %>%
    mutate(RMSE=tuningResults) %>%
    group_by(range, nugget, mtry, min.node.size) %>%
    summarize(RMSE=mean(RMSE)) %>%
    ungroup() %>%
    dplyr::slice(which.min(RMSE))
  spatial_RF_preds <- get_RF_preds(rng=bestTune$range,
                                   nug=bestTune$nugget,
                                   Mt=bestTune$mtry,
                                   mns=bestTune$min.node.size)
  RMSE_mat_spatial[2, cv] <- rmse_vec(validData$PM25, spatial_RF_preds)
  
  RF <- ranger(PM25~., data=trainData,
               importance="impurity")
  RF_preds <- predict(RF, data=validData)$predictions
  RMSE_mat[2, cv] <- rmse_vec(validData$PM25,
                              RF_preds)
  
  ####################
  ## Fit MLP Models ##
  ####################
  
  ## Determine Grid of Tuning Parameters
  tuningGrid <- expand.grid(range=sr.seq,
                            nugget=pct.spatial,
                            fold=1:nfolds,
                            hidden_units=seq(3,12))
  
  ## Function to get RF RMSE for tuning
  get_MLP_RMSE <- function(rng, nug, fld, hu){
    ## Transform to independent
    trndata <- get_rsplit(folds, index=fld) %>%
      analysis(.)
    trnlocs <- get_rsplit(folds, index=fld) %>%
      analysis(.) %>%
      select(X, Y) %>%
      as.matrix()
    vlddata <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(-PM25)
    vldlocs <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(X, Y) %>%
      as.matrix()
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores=5)
    
    ## Fit Model
    mlp_mod <- mlp(hidden_units=hu) %>%
      set_engine("nnet") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    mlp_preds <- predict(mlp_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    mlp_preds <- back_transform_to_spatial(preds=mlp_preds, indData)
    
    ## Retain Assessment
    rmseVal <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      pull(PM25) %>%
      rmse_vec(., mlp_preds)
    return(rmseVal)
  }
  
  ## Function to get RF preds
  get_MLP_preds <- function(rng, nug, hu){
    ## Transform to independent
    trndata <- trainData
    trnlocs <- trainLocs
    vlddata <- validData %>%
      select(-PM25)
    vldlocs <- validLocs
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    mlp_mod <- mlp(hidden_units=hu) %>%
      set_engine("nnet") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    mlp_preds <- predict(mlp_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    mlp_preds <- back_transform_to_spatial(preds=mlp_preds, indData)
    
    ## Retain Preds
    return(mlp_preds)
  }
  
  ## Run the Function for all tuning parameters
  tuningResults <- mclapply(1:nrow(tuningGrid), FUN=function(ind){
    return(get_MLP_RMSE(rng=tuningGrid$range[ind],
                        nug=tuningGrid$nugget[ind],
                        fld=tuningGrid$fold[ind],
                        hu=tuningGrid$hidden_units[ind]))
  }, mc.cores=detectCores()) %>%
    do.call("c", args=.)
  
  bestTune <- tuningGrid %>%
    mutate(RMSE=tuningResults) %>%
    group_by(range, nugget, hidden_units) %>%
    summarize(RMSE=mean(RMSE)) %>%
    ungroup() %>%
    dplyr::slice(which.min(RMSE))
  spatial_MLP_preds <- get_MLP_preds(rng=bestTune$range,
                                     nug=bestTune$nugget,
                                     hu=bestTune$hidden_units)
  RMSE_mat_spatial[3, cv] <- rmse_vec(validData$PM25, spatial_MLP_preds)
  
  MLP <- nnet(PM25~., data=trainData,
              linout=TRUE,
              size=bestTune$hidden_units)
  MLP_preds <- c(predict(MLP, newdata=validData))
  RMSE_mat[3, cv] <- rmse_vec(validData$PM25, 
                              MLP_preds)
  
  #########################
  ## Fit Boosting Models ##
  #########################
  
  tuningGrid <- expand.grid(range=sr.seq,
                            nugget=pct.spatial,
                            fold=1:nfolds,
                            mtry=1:8,
                            min.node.size=seq(5,25,by=5))
  
  get_GBM_RMSE <- function(rng, nug, fld, Mt, mns){
    ## Transform to independent
    trndata <- get_rsplit(folds, index=fld) %>%
      analysis(.)
    trnlocs <- get_rsplit(folds, index=fld) %>%
      analysis(.) %>%
      select(X, Y) %>%
      as.matrix()
    vlddata <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(-PM25)
    vldlocs <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(X, Y) %>%
      as.matrix()
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    gbm_mod <- boost_tree(mtry=Mt,
                          min_n=mns) %>%
      set_engine("xgboost") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    gbm_preds <- predict(gbm_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    gbm_preds <- back_transform_to_spatial(preds=gbm_preds, indData)
    
    ## Retain Assessment
    rmseVal <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      pull(PM25) %>%
      rmse_vec(., gbm_preds)
    return(rmseVal)
  }
  
  ## Function to get RF preds
  get_GBM_preds <- function(rng, nug, Mt, mns){
    ## Transform to independent
    trndata <- trainData
    trnlocs <- trainLocs
    vlddata <- validData %>%
      select(-PM25)
    vldlocs <- validLocs
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    gbm_mod <- boost_tree(mtry=Mt,
                          min_n=mns) %>%
      set_engine("xgboost") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    gbm_preds <- predict(gbm_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    gbm_preds <- back_transform_to_spatial(preds=gbm_preds, indData)
    
    ## Retain Preds
    return(gbm_preds)
  }
  
  ## Run the Function for all tuning parameters
  tuningResults <- mclapply(1:nrow(tuningGrid), FUN=function(ind){
    return(get_GBM_RMSE(rng=tuningGrid$range[ind],
                        nug=tuningGrid$nugget[ind],
                        fld=tuningGrid$fold[ind],
                        Mt=tuningGrid$mtry[ind],
                        mns=tuningGrid$min.node.size[ind]))
  }, mc.cores=detectCores()) %>%
    do.call("c", args=.)
  
  bestTune <- tuningGrid %>%
    mutate(RMSE=tuningResults) %>%
    group_by(range, nugget, mtry, min.node.size) %>%
    summarize(RMSE=mean(RMSE)) %>%
    ungroup() %>%
    dplyr::slice(which.min(RMSE))
  spatial_GBM_preds <- get_GBM_preds(rng=bestTune$range,
                                     nug=bestTune$nugget,
                                     Mt=bestTune$mtry,
                                     mns=bestTune$min.node.size)
  RMSE_mat_spatial[4, cv] <- rmse_vec(validData$PM25, spatial_GBM_preds)
  
  GBM <- boost_tree(mtry=bestTune$mtry,
                    min_n=bestTune$min.node.size) %>%
    set_engine("xgboost") %>%
    set_mode("regression") %>%
    fit(formula=PM25~., data=trainData)
  GBM_preds <- predict(GBM, new_data=validData) %>% pull(.pred)
  RMSE_mat[4, cv] <- rmse_vec(validData$PM25, 
                              GBM_preds)
  
  #####################
  ## Fit BART Models ##
  #####################
  
  tuningGrid <- expand.grid(range=sr.seq,
                            nugget=pct.spatial,
                            fold=1:nfolds)
  
  get_BART_RMSE <- function(rng, nug, fld){
    ## Transform to independent
    trndata <- get_rsplit(folds, index=fld) %>%
      analysis(.)
    trnlocs <- get_rsplit(folds, index=fld) %>%
      analysis(.) %>%
      select(X, Y) %>%
      as.matrix()
    vlddata <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(-PM25)
    vldlocs <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(X, Y) %>%
      as.matrix()
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    bart_mod <- bart() %>%
      set_engine("dbarts") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    bart_preds <- predict(bart_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    bart_preds <- back_transform_to_spatial(preds=bart_preds, indData)
    
    ## Retain Assessment
    rmseVal <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      pull(PM25) %>%
      rmse_vec(., bart_preds)
    return(rmseVal)
  }
  
  ## Function to get RF preds
  get_BART_preds <- function(rng, nug){
    ## Transform to independent
    trndata <- trainData
    trnlocs <- trainLocs
    vlddata <- validData %>%
      select(-PM25)
    vldlocs <- validLocs
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    bart_mod <- bart() %>%
      set_engine("dbarts") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    bart_preds <- predict(bart_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    bart_preds <- back_transform_to_spatial(preds=bart_preds, indData)
    
    ## Retain Preds
    return(bart_preds)
  }
  
  ## Run the Function for all tuning parameters
  tuningResults <- mclapply(1:nrow(tuningGrid), FUN=function(ind){
    return(get_BART_RMSE(rng=tuningGrid$range[ind],
                         nug=tuningGrid$nugget[ind],
                         fld=tuningGrid$fold[ind]))
  }, mc.cores=detectCores()) %>%
    do.call("c", args=.)
  
  bestTune <- tuningGrid %>%
    mutate(RMSE=tuningResults) %>%
    group_by(range, nugget) %>%
    summarize(RMSE=mean(RMSE)) %>%
    ungroup() %>%
    dplyr::slice(which.min(RMSE))
  spatial_BART_preds <- get_BART_preds(rng=bestTune$range,
                                       nug=bestTune$nugget)
  RMSE_mat_spatial[5, cv] <- rmse_vec(validData$PM25, spatial_BART_preds)
  
  BART <- bart() %>%
    set_engine("dbarts") %>%
    set_mode("regression") %>%
    fit(formula=PM25~., data=trainData)
  BART_preds <- predict(BART, new_data=validData) %>% pull(.pred)
  RMSE_mat[5, cv] <- rmse_vec(validData$PM25, 
                              BART_preds)
  
  #####################
  ## Fit KNN Models ##
  #####################
  
  tuningGrid <- expand.grid(range=sr.seq,
                            nugget=pct.spatial,
                            fold=1:nfolds,
                            nn=seq(3,25))
  
  get_KNN_RMSE <- function(rng, nug, fld, nn){
    ## Transform to independent
    trndata <- get_rsplit(folds, index=fld) %>%
      analysis(.)
    trnlocs <- get_rsplit(folds, index=fld) %>%
      analysis(.) %>%
      select(X, Y) %>%
      as.matrix()
    vlddata <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(-PM25)
    vldlocs <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      select(X, Y) %>%
      as.matrix()
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    knn_mod <- nearest_neighbor(neighbors=nn) %>%
      set_engine("kknn") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    knn_preds <- predict(knn_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    knn_preds <- back_transform_to_spatial(preds=knn_preds, indData)
    
    ## Retain Assessment
    rmseVal <- get_rsplit(folds, index=fld) %>%
      assessment(.) %>%
      pull(PM25) %>%
      rmse_vec(., knn_preds)
    return(rmseVal)
  }
  
  ## Function to get RF preds
  get_knn_preds <- function(rng, nug, nn){
    ## Transform to independent
    trndata <- trainData
    trnlocs <- trainLocs
    vlddata <- validData %>%
      select(-PM25)
    vldlocs <- validLocs
    indData <- transform_to_ind(formula=PM25~.,
                                trainData=trndata,
                                trainLocs=trnlocs,
                                testData=vlddata,
                                testLocs=vldlocs,
                                MaternParams=c(rng, nug),
                                ncores = 5)
    
    ## Fit Model
    knn_mod <- nearest_neighbor(neighbors=nn) %>%
      set_engine("kknn") %>%
      set_mode("regression") %>%
      fit(formula=y~., data=indData$trainData)
    
    ## Get Predictions
    knn_preds <- predict(knn_mod, new_data=indData$testData) %>%
      pull(.pred)
    
    ## Back Transform
    knn_preds <- back_transform_to_spatial(preds=knn_preds, indData)
    
    ## Retain Preds
    return(knn_preds)
  }
  
  ## Run the Function for all tuning parameters
  tuningResults <- mclapply(1:nrow(tuningGrid), FUN=function(ind){
    return(get_KNN_RMSE(rng=tuningGrid$range[ind],
                        nug=tuningGrid$nugget[ind],
                        fld=tuningGrid$fold[ind],
                        nn=tuningGrid$nn[ind]))
  }, mc.cores=detectCores()) %>%
    do.call("c", args=.)
  
  bestTune <- tuningGrid %>%
    mutate(RMSE=tuningResults) %>%
    group_by(range, nugget, nn) %>%
    summarize(RMSE=mean(RMSE)) %>%
    ungroup() %>%
    dplyr::slice(which.min(RMSE))
  spatial_KNN_preds <- get_knn_preds(rng=bestTune$range,
                                     nug=bestTune$nugget,
                                     nn=bestTune$nn)
  RMSE_mat_spatial[6, cv] <- rmse_vec(validData$PM25, spatial_KNN_preds)
  
  KNN <- nearest_neighbor(neighbors=bestTune$nn) %>%
    set_engine("kknn") %>%
    set_mode("regression") %>%
    fit(formula=PM25~., data=trainData)
  KNN_preds <- predict(KNN, new_data=validData) %>% pull(.pred)
  RMSE_mat[6, cv] <- rmse_vec(validData$PM25, 
                              KNN_preds)
  
  
  ######################
  ## Fit GeoRF Models ##
  ######################
  
  ## Determine Grid of Tuning Parameters
  # tuningGrid <- expand.grid(fold=1:nfolds,
  #                           mtry=1:8,
  #                           min.node.size=seq(5,25,by=5))
  # 
  # ## Function to get RF RMSE for tuning
  # get_geoRF_RMSE <- function(fld, Mt, mns){
  #   ## Transform to Augmented Dataframe
  #   trndata <- get_rsplit(folds, index=fld) %>%
  #     analysis(.)
  #   trnlocs <- get_rsplit(folds, index=fld) %>%
  #     analysis(.) %>%
  #     select(X, Y) %>%
  #     as.matrix()
  #   vlddata <- get_rsplit(folds, index=fld) %>%
  #     assessment(.) %>%
  #     select(-PM25)
  #   vldlocs <- get_rsplit(folds, index=fld) %>%
  #     assessment(.) %>%
  #     select(X, Y) %>%
  #     as.matrix()
  #   indData <- Geodata(trndata, vlddata, trnlocs, vldlocs, 5)
  #   
  #   ## Fit Model
  #   rf_mod <- ranger(PM25~.,
  #                    data=indData$trainData,
  #                    num.trees=500,
  #                    mtry=Mt,
  #                    min.node.size=mns)
  #   
  #   ## Get Predictions
  #   rf_preds <- predict(rf_mod, data=indData$testData)$predictions
  #   
  #   ## Retain Assessment
  #   rmseVal <- get_rsplit(folds, index=fld) %>%
  #     assessment(.) %>%
  #     pull(PM25) %>%
  #     rmse_vec(., rf_preds)
  #   return(rmseVal)
  # }
  # 
  # ## Function to get RF preds
  # get_geoRF_preds <- function(Mt, mns){
  #   ## Transform to independent
  #   trndata <- trainData
  #   trnlocs <- trainLocs
  #   vlddata <- validData %>%
  #     select(-PM25)
  #   vldlocs <- validLocs
  #   indData <- Geodata(trndata, vlddata, trnlocs, vldlocs, 5)
  #   
  #   ## Fit Model
  #   rf_mod <- ranger(PM25~.,
  #                    data=indData$trainData,
  #                    num.trees=500,
  #                    mtry=Mt,
  #                    min.node.size=mns)
  #   
  #   ## Get Predictions
  #   rf_preds <- predict(rf_mod, data=indData$testData)$predictions
  #   
  #   ## Retain Preds
  #   return(rf_preds)
  # }
  # 
  # ## Run the Function for all tuning parameters
  # tuningResults <- mclapply(1:nrow(tuningGrid), FUN=function(ind){
  #   return(get_geoRF_RMSE(fld=tuningGrid$fold[ind],
  #                      Mt=tuningGrid$mtry[ind],
  #                      mns=tuningGrid$min.node.size[ind]))
  # }, mc.cores=detectCores()) %>%
  #   do.call("c", args=.)
  # 
  # bestTune <- tuningGrid %>%
  #   mutate(RMSE=tuningResults) %>%
  #   group_by(mtry, min.node.size) %>%
  #   summarize(RMSE=mean(RMSE)) %>%
  #   ungroup() %>%
  #   dplyr::slice(which.min(RMSE))
  # spatial_geoRF_preds <- get_geoRF_preds(Mt=bestTune$mtry,
  #                                  mns=bestTune$min.node.size)
  # RMSE_mat_spatial[7, cv] <- rmse_vec(validData$PM25, spatial_geoRF_preds)
  # 
  # RF <- ranger(PM25~., data=trainData,
  #              importance="impurity")
  # RF_preds <- predict(RF, data=validData)$predictions
  # RMSE_mat[7, cv] <- rmse_vec(validData$PM25,
  #                             RF_preds)
  # 
  ## Increment Progess Bar
  setTxtProgressBar(pb,cv)
  
} #END CV For Loop
close(pb)

rownames(RMSE_mat) <- rownames(RMSE_mat_spatial) <-
  c("LM", "RF", "SLP", "Boost", "BART", "KNN", "GeoRF")
colnames(RMSE_mat) <- colnames(RMSE_mat_spatial) <-
  paste0("Fold",1:ncol(cvFolds))

save(file="./CVResults.RData", list=c("RMSE_mat", "RMSE_mat_spatial"))




