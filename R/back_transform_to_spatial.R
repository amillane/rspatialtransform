
#' Title
#'
#' @param preds A vector of predictions from the machine learning model
#' @param transformObj The object outputted IndData after running transform_to_ind
#'
#' @returns A vector of predictions that have been transformed back to the original spatial scale
#' @export
#'
#' @title Back Transform to Spatial
#' @description This function takes a vector of predictions from a machine learning model that have been transformed to an independent scale and transforms them back to the original spatial scale.
back_transform_to_spatial <- function(preds, transformObj){
  
  spatialPreds <- preds*sapply(transformObj$backTransformInfo, function(x){x$w})+
    sapply(transformObj$backTransformInfo, function(x){x$backTrans})
  return(spatialPreds)
  
}
