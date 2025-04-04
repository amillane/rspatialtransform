predict_NN_Matern <- function(NNMaternModel,predlocs,newdata=NULL){
  
  ## Errors
  if(is.null(newdata) & length(NNMaternModel$coefTable$Estimate)>1){
    stop(paste("MaternModel indicates the use of covariates.",
               "Please supply covariates at prediction locations via newdata"))
  }
  
  ## Determine prediction X matrix
  if(is.null(newdata)){
    predModelMatrix <- model.matrix(predlocs~1)
  } else {
    predModelMatrix <- model.matrix(NNMaternModel$frm,data=newdata)
  }
  
  ## Get prediction point by point
  getPred <- function(ind){
    nn <- order(rdist(matrix(predlocs[ind,],ncol=ncol(NNMaternModel$locs)),
                      NNMaternModel$locs))[1:NNMaternModel$n.neighbors]
    nnLocs <- rbind(matrix(predlocs[ind,], ncol=ncol(NNMaternModel$locs)),
                    matrix(NNMaternModel$locs[nn,], ncol=ncol(NNMaternModel$locs)))
    nnR <- NNMaternModel$nugget*diag(nrow(nnLocs))+
      (1-NNMaternModel$nugget)*Matern(rdist(nnLocs), nu=NNMaternModel$nu, alpha=NNMaternModel$decay)
    pred <- predModelMatrix[ind,]%*%NNMaternModel$coefTable$Estimate +
      nnR[1,-1]%*%solve(nnR[-1,-1])%*%(NNMaternModel$response[nn]-
                                         matrix(NNMaternModel$X[nn,],ncol=ncol(NNMaternModel$X))%*%
                                         NNMaternModel$coefTable$Estimate)
    se.pred <- NNMaternModel$sigma2*(1-nnR[1,-1]%*%solve(nnR[-1,-1])%*%nnR[-1,1])
    return(list(pred=pred, se.pred=se.pred))
  }
  allPreds <- lapply(1:nrow(predlocs), getPred)
  
  return(data.frame(predlocs=predlocs,pred=sapply(allPreds, function(v){v$pred}),
                    se=sapply(allPreds, function(v){v$se.pred})))
  
}
