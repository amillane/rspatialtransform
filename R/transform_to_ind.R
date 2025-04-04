##
## Function to transform to independence
##


#' Title
#'
#' @param formula An object of class "formula" describing the model to be decorrelated.
#' @param trainData An object of class data.frame containing the training data with the response variable provided.
#' @param trainLocs A matrix object containing the coordinates of the training data. The dimensions should be nx2.
#' @param testData An object of class data.frame containing the data to be predicted.
#' @param testLocs A matrix object containing the coordinates of the test data. The dimensions should be nx2.
#' @param MaternParams A vector of two parameters: range and nugget. Range represents how fast the correlation decays with distance and nugget represents the variability in one location. The default is NULL (rng, nug) where the range and nugget parameter are estimated automatically.
#' @param smoothness The smoothness parameter, which controls the smoothness of the function. The default is 1/2 which results in an exponential kernel.
#' @param M The number of neighbors to consider when creating a correlation matrix for each individual observation. The default is 30.
#' @param ncores The number of cores to parallelize the decorrelation process.
#'
#' @returns list
#' @export
#'
#' @title Transform Spatial Data to Independent Data
#' @description This function takes a formula, training data, and test data and transforms the data to an independent scale. The function returns a list containing the transformed training data, transformed test data, range, nugget, number of neighbors, and back transformation information.
#'
#'
transform_to_ind <- function(formula,
                             trainData,
                             trainLocs,
                             testData, #Don't include response
                             testLocs,
                             MaternParams=NULL, #Either null or a 2 vector of (rng, nug)
                             smoothness=1/2,
                             M = 30, #num neighbors
                             ncores=parallel::detectCores()-5){
  
  ######################################
  ## Figure out the nearest neighbors ##
  ######################################
  nnList <- self_mkNNIndx(trainLocs, m=M)
  
  #################################################
  ## Estimate a range and nugget if not provided ##
  #################################################
  if(is.null(MaternParams)){
    mFit <- fit_NN_Matern(formula, data=trainData, locs=trainLocs, nu=smoothness,
                          NearNeighs=nnList, num.cores=ncores)
    range <- 1/mFit$decay
    nugget <- mFit$nugget
  } else {
    range <- MaternParams[1]
    nugget <- MaternParams[2]
  }
  
  
  ################################
  ## Transform the Training Set ##
  ################################
  
  ## Define X and y matrices
  Xtrain <- stats::model.matrix(formula, data=trainData)
  ytrain <- as.matrix(trainData[,all.vars(formula)[1]], ncol=1)
  
  ## Apply decorrelating transform by location
  indData <- parallel::mclapply(1:nrow(Xtrain), FUN=function(idx){
    if(idx==1){
      y <- ytrain[idx]
      w <- 1
      X <- Xtrain[idx,] / sqrt(w)
    } else if(idx==2){
      D <- fields::rdist(trainLocs[1:idx,])
      R <- (1-nugget)*fields::Matern(D, nu=1/2, range=range) +
        nugget*diag(nrow(D))
      w <- as.numeric(1-R[1,-1]%*%solve(R[-1,-1])%*%R[-1,1])
      X <- (t(Xtrain[idx,]) - (R[1,-1]%*%solve(R[-1,-1])%*%Xtrain[nnList[[idx]],])) / sqrt(w)
      y <- (ytrain[idx]-R[1,-1]%*%solve(R[-1,-1])%*%(ytrain[nnList[[idx]]]))/sqrt(w)
    } else {
      D <- fields::rdist(trainLocs[c(idx,nnList[[idx]]),])
      R <- (1-nugget)*fields::Matern(D, nu=1/2, range=range) +
        nugget*diag(nrow(D))
      w <- as.numeric(1-R[1,-1]%*%solve(R[-1,-1])%*%R[-1,1])
      X <- (t(Xtrain[idx,]) - (R[1,-1]%*%solve(R[-1,-1])%*%Xtrain[nnList[[idx]],])) / sqrt(w)
      y <- (ytrain[idx]-R[1,-1]%*%solve(R[-1,-1])%*%(ytrain[nnList[[idx]]]))/sqrt(w)
    }
    
    return(list(y=y, X=X, w=w))
  }, mc.cores=ncores) # End mclapply()
  
  ## Apply decorrelating transform to test data
  Xtest <- stats::model.matrix(~ ., data=testData)
  indTestData <- parallel::mclapply(1:nrow(Xtest), FUN=function(idx){
    D <- fields::rdist(matrix(testLocs[idx,], nrow=1), trainLocs)
    theNeighbors <- order(D)[1:M]
    R <- fields::rdist(rbind(testLocs[idx,],trainLocs[theNeighbors,]))
    R <- nugget*diag(M+1)+(1-nugget)*fields::Matern(R, range=range, smoothness=smoothness)
    R12 <- R[1,-1]%*%chol2inv(chol(R[-1,-1]))
    w <- as.numeric(1-R12%*%R[-1,1])
    X <- (t(Xtest[idx,])-R12%*%Xtrain[theNeighbors,])/sqrt(w)
    return(list(backTrans=R12%*%matrix(ytrain[theNeighbors,], ncol=1), X=X,
                w=w))
  }, mc.cores=ncores)
  
  ## Return transformed data
  outList <- list(trainData=data.frame(y=do.call(rbind, lapply(indData, function(x){x$y})),
                                       do.call(rbind, lapply(indData, function(x){x$X}))),
                  testData=data.frame(do.call(rbind, lapply(indTestData, function(x){x$X}))),
                  range=range,
                  nugget=nugget,
                  M=M,
                  formula=formula,
                  backTransformInfo=lapply(indTestData,function(x){x$X<-NULL
                  return(x)}))
  return(outList)
  
  
} # End spatial_to_ind function



