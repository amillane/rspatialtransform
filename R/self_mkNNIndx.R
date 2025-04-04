

self_mkNNIndx <- function(coords, m, n.omp.threads=1){
  
  get.n.indx <- function(i, m){
    i <- i-1
    if(i == 0){
      return(NA)
    }else if(i < m){
      n.indx.i <- i/2*(i-1)
      m.i <- i
      return((n.indx.i+1):((n.indx.i+1)+i-1))
    }else{
      n.indx.i <- m/2*(m-1)+(i-m)*m
      m.i <- m
      return((n.indx.i+1):((n.indx.i+1)+m-1))
    }
  }
  
  n <- nrow(coords)
  nIndx <- (1+m)/2*m+(n-m-1)*m
  nnIndx <- rep(0, nIndx)
  nnDist <- rep(0, nIndx)
  nnIndxLU <- matrix(0, n, 2)
  
  n <- as.integer(n)
  m <- as.integer(m)
  coords <- as.double(coords)
  nnIndx <- as.integer(nnIndx)
  nnDist <- as.double(nnDist)
  nnIndxLU <- as.integer(nnIndxLU)
  n.omp.threads <- as.integer(n.omp.threads)
  
  out <- .Call("mkNNIndx", n, m, coords, nnIndx, nnDist, nnIndxLU, n.omp.threads,
               PACKAGE= 'spNNGP')
  
  n.indx <- as.integer(nnIndx)
  
  n.indx.list <- vector("list", n)
  n.indx.list[1] <- NA
  for(i in 2:n){
    n.indx.list[[i]] <- n.indx[get.n.indx(i, m)]+1
  }
  n.indx.list
}

