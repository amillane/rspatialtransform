
# rspatialtransform

<!-- badges: start -->
<!-- badges: end -->

The goal of rspatialtransform is to transform spatially correlated data to independent data to be used in machine learning models. 

## Installation

You can install the development version of rspatialtransform from [GitHub](https://github.com/) with:

``` r
# install.packages('devtools')
devtools::install_github('amillane/rspatialtransform')
```

## Example

This is a basic example which shows you how to implement the transformation:

``` r
## usage example
library(rspatialtransform)

transform_to_ind <- function(y~.,
                             trainData,
                             trainLocs,
                             testData, #Don't include response
                             testLocs,
                             MaternParams=(range,nug), 
                             smoothness=1/2,
                             M = 30, #num neighbors
                             ncores=detectCores()-10)
```

