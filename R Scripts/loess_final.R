library(aspline)
library(tidyverse)
library(splines2)
library(pracma)
library(bootstrap)
library(Bolstad2)
library(rootSolve)
library(PerMallows)
library(textir)
loess_wrapper_extrapolate <- function(x, y, span.vals = seq(0.20, 0.65, by = 0.05), folds = 3){
  # Do model selection using mean absolute error, which is more robust than squared error.
  mean.abs.error <- numeric(length(span.vals))
  
  # Quantify error for each span, using CV
  loess.model <- function(x, y, span){
    loess(y ~ x, span = span, control=loess.control(surface="direct"))
  }
  
  loess.predict <- function(fit, newdata) {
    predict(fit, newdata = newdata)
  }
  
  span.index <- 0
  for (each.span in span.vals) {
    span.index <- span.index + 1
    y.hat.cv <- crossval(x, y, theta.fit = loess.model, theta.predict = loess.predict, span = each.span, ngroup = folds)$cv.fit
    non.empty.indices <- !is.na(y.hat.cv)
    mean.abs.error[span.index] <- mean(abs(y[non.empty.indices] - y.hat.cv[non.empty.indices]))
  }
  
  # find the span which minimizes error
  best.span <- span.vals[which.min(mean.abs.error)]
  
  # fit and return the best model
  best.model <- loess(y ~ x, span = best.span, control=loess.control(surface="direct"))
  return(list(best.model,best.span))
}

loess_stuff <- function(x,y,folds=5){
  best = loess_wrapper_extrapolate(x,y, fold = folds)
  model = best[[1]]
  span = best[[2]]
  Loess_x_grid = seq(0,86399,by=1)
  
  FinalLoessFit <- predict(model,Loess_x_grid)
  
  Loess_Total_Area_Trapz <- trapz(Loess_x_grid, FinalLoessFit)
  
  Loess_Total_Area_Simpsons <- sintegral(Loess_x_grid, FinalLoessFit)$int
  
  return (list(Loess_Total_Area_Trapz,Loess_Total_Area_Simpsons,FinalLoessFit,span))
}

loess_simple <- function(x,y,span){
	
	model <- loess(y ~ x, span = span, control=loess.control(surface="direct"))
	Loess_x_grid = seq(0,86399,by=1)
  	
  	FinalLoessFit <- predict(model,Loess_x_grid)
  
  	Loess_Total_Area_Trapz <- trapz(Loess_x_grid, FinalLoessFit)
  
  	Loess_Total_Area_Simpsons <- sintegral(Loess_x_grid, FinalLoessFit)$int
  
  	return (list(Loess_Total_Area_Trapz,Loess_Total_Area_Simpsons,FinalLoessFit,span))
}