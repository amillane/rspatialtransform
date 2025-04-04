## Fit Spatial NN Model using ML
fit_NN_Matern <- function(formula,locs,nu,gridsize=15,NearNeighs,
                          num.cores=parallel::detectCores(),data=NULL){
  
  ## Assign variables
  X <- stats::model.matrix(formula,data=data)
  y <- matrix(stats::model.frame(formula,data=data)[,1],ncol=1)
  n <- nrow(X)
  if(length(gridsize)==1){
    sr.gridsize <- gridsize
    pct.gridsize <- gridsize
  } else {
    sr.gridsize <- gridsize[1]
    pct.gridsize <- gridsize[2]
  }
  
  ## Order the locations
  if(is.null(dim(locs))){
    locs <- matrix(locs, ncol=1)
  }
  ord <- GPvecchia::order_maxmin_exact(locs)
  locs <- locs[ord,]
  y <- matrix(y[ord], ncol=1)
  X <- matrix(X[ord,], ncol=ncol(X))
  
  ## Create a Sequence for Spatial Range
  D <- fields::rdist(locs[sample(n, size=min(n,500)),])
  max.dist <- max(D)
  min.dist <- max(apply(D,1,function(x){sort(x)[2]}))
  upperbound.decay <- 1/fields::Matern.cor.to.range(min.dist,nu=nu,cor.target=0.05)
  lowerbound.decay <- 1/fields::Matern.cor.to.range(max.dist,nu=nu,cor.target=0.95)
  #c(lowerbound.decay,upperbound.decay)
  sr.seq <- seq(lowerbound.decay,upperbound.decay,length=sr.gridsize)
  
  ## Create a Sequence for %Spatial
  pct.spatial <- seq(0,.99,length=pct.gridsize)
  
  ## Expand pct and spatial range grid
  pct.sr.grid <- expand.grid(pct.spatial,sr.seq)
  
  ## Parse it out into a list for parallel processing
  aMw.list <- vector('list',nrow(pct.sr.grid))
  for(i in 1:length(aMw.list)){
    aMw.list[[i]] <- list(alpha=pct.sr.grid[i,2],omega=1-pct.sr.grid[i,1])
  }
  
  ## Function for calculating likelihoods that can be run in parallel
  getLL <- function(x){
    ## Transform to ind y & x
    getYX <- function(ind){
      if(ind==1){
        return(list(y_IID=y[ind],X_IID=X[ind,]))
      } else {
        R <- x$omega*diag(1+min(ind-1, length(NearNeighs[[ind]]))) +
          (1-x$omega)*fields::Matern(fields::rdist(locs[c(ind, NearNeighs[[ind]]),]), alpha=x$alpha)
        RiRn_inv <- R[1,-1]%*%chol2inv(chol(R[-1,-1]))
        Xtrans <- X[ind,]-RiRn_inv%*%X[NearNeighs[[ind]],]
        ytrans <- y[ind]-RiRn_inv%*%y[NearNeighs[[ind]]]
        return(list(y_IID=ytrans,X_IID=Xtrans))
      }
    }
    yx <- lapply(1:n, getYX)
    iidY <- matrix(sapply(yx, function(v){v$y_IID}), ncol=1)
    iidX <- lapply(yx, function(v){v$X_IID}) %>% do.call(rbind,.)
    
    ## Find bhat
    bhat <- solve(t(iidX)%*%iidX)%*%t(iidX)%*%iidY
    
    ## Find sig2hat
    sig2hat <- sum((iidY-iidX%*%bhat)^2)/n
    
    ## Get ll
    return(list(ll=sum(dnorm(iidY, iidX%*%bhat, sqrt(sig2hat),log=TRUE)),
                bhat=bhat,
                bse=diag(solve(t(iidX)%*%iidX)),
                sigma2=sig2hat,
                nugget=x$omega,
                decay=x$alpha
    ))
  }
  
  ## Apply likelihood function to each combo
  ll.list <- parallel::mclapply(aMw.list, getLL, mc.cores=num.cores)
  
  ## Find max(ll)
  all.ll <- sapply(ll.list,function(x){return(x$ll)})
  max.ll <- which.max(all.ll)
  ll.list <- ll.list[[max.ll]]
  coef.table <- data.frame(Estimate=ll.list$bhat,StdErr=sqrt(ll.list$bse*ll.list$sigma2),
                           TestStat=ll.list$bhat/sqrt(ll.list$bse*ll.list$sigma2),
                           PVal2Sided=2*pnorm(abs(ll.list$bhat/sqrt(ll.list$bse*ll.list$sigma2)),lower.tail=FALSE))
  rownames(coef.table) = colnames(X)
  
  ## Return Info
  return(list(coefTable=coef.table,sigma2=ll.list$sigma2,nugget=ll.list$nugget,
              decay=ll.list$decay,loglike=ll.list$ll,
              response=y,locs=locs,nu=nu,X=X,frm=formula,
              n.neighbors=length(NearNeighs[[n]])))
}
