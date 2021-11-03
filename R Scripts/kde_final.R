library(evmix)
library(ks)
kde_function <- function(x,TotalArea,kernel = "epanechnikov"){
  test<-density(x = x, kernel = kernel, from=0, to=86399,n=86400)
  Yhat<-(TotalArea*test$y)/sum(test$y)
  return (Yhat)
}

kde_function_custom <- function(x,TotalArea,kernel = "epanechnikov",from_=0, to=86400){
  test<-density(x = x, kernel = kernel, from= from_, to= to,n=86400)
  Yhat<-(TotalArea*test$y)/sum(test$y)
  return (list(Yhat,test$bw))
}

kde_function_custom_bw <- function(x,TotalArea,bw,kernel = "epanechnikov",from_=0, to=86400){
  test<-density(x = x, kernel = kernel, from= from_, to= to,n=86400,bw= bw[1])
  Yhat<-(TotalArea*test$y)/sum(test$y)
  return (Yhat)
}

kde_function_noScaling <- function(x,kernel = "epanechnikov"){
  test<-density(x = x, kernel = kernel, from=0, to=86399,n=86400)
  return (test$y)
}

kde_function_custom_noScaling <- function(x,kernel = "epanechnikov",from_=0, to=86400){
  test<-density(x = x, kernel = kernel, from= from_, to= to,n=86400)
  return (test$y)
}

kde_function_boundary <- function(x,TotalArea,kernel = "epanechnikov"){
  test<- kde.boundary(x = x, xmin=0,xmax=86400)
  return (test)
}

