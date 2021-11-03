library(aspline)
library(tidyverse)
library(splines2)
library(pracma)
library(bootstrap)
library(Bolstad2)
library(rootSolve)
library(PerMallows)
library(textir)

# velocity1 = read.csv("velocity1.csv")
# velocity2 = read.csv("velocity2.csv")
# data = velocity2
# x <- data$Hour
# y <- data$Velocity

# Aspline_Area_Proportions function

Aspline_Area_Proportions <- function(x,y,degree,k=40){
  knots <- seq(min(x), max(x), length = k + 2)[-c(1, k + 2)]
  pen <- 10 ^ seq(-2, 2, 0.25)
  x_seq <- seq(0,86399,1)
  aridge <- aridge_solver(x, y, knots, pen, degree = degree)
  a_fit <- lm(y ~ bSpline(x, knots = aridge$knots_sel[[which.min(aridge$ebic)]], degree = degree))
  X_seq <- bSpline(x_seq, knots = aridge$knots_sel[[which.min(aridge$ebic)]], intercept = TRUE, degree = degree)
  a_basis <- (X_seq %*% diag(coef(a_fit))) %>%
    as.data.frame() %>%
    mutate(x = x_seq) %>%
    reshape2::melt(id.vars = "x", variable.name = "spline_n", value.name = "y") %>%
    as_tibble() %>%
    filter(y != 0)
  a_predict <- data_frame(x = x_seq, pred = predict(a_fit, data.frame(x = x_seq)))
  
  plot <- ggplot() +
    geom_point(data = data, aes(x, y), shape = 1) +
    geom_line(data = a_predict, aes(x, pred, colour="red"), size = 0.9) +
    geom_vline(xintercept = aridge$knots_sel[[which.min(aridge$ebic)]], color = 'blue',linetype="dotted", size = 0.75) +
    scale_color_hue(labels = c("Piecewise Spline Fits")) + 
    theme(legend.position = "top") +
    guides(color=guide_legend("")) +
    ylab("Velocity") +
    xlab("Time of the day (in seconds)")
  
  final_knots <- aridge$knots_sel[[which.min(aridge$ebic)]] # Knots
  
  # Calulating Analytical area under the piecewise Aspline-degree 1
  
  if (degree==1){
    x_points = c(min(x),final_knots,max(x))
    intercepts = rep(0, length(final_knots)+1)
    slopes = rep(0, length(final_knots)+1)
    
    for (i in 1:length(intercepts)) {
      df <- subset(a_predict, x >= x_points[i] & x <= x_points[i+1])
      tempmodel_coefs <- coef(lm(pred ~ x, data=df))
      intercepts[i] = tempmodel_coefs[1]
      slopes[i] = tempmodel_coefs[2]
    }
    
    areas = rep(0, length(final_knots)+1)
    
    for (i in 1:length(areas)) {
      func = function(x,k=i) {return(slopes[k]*x + intercepts[k])}
      areas[i] = integral(func, x_points[i], x_points[i+1])
    }
    
    relative_proportions <- rep(0, length(final_knots)+1)
    
    for (i in 1:length(relative_proportions)) {
      relative_proportions[i] = areas[i]/sum(areas)
    }
    
    return_list <- list(relative_proportions,final_knots,plot,intercepts,slopes,sum(areas),a_predict)
    
  }
  
  # Calulating Analytical area under the piecewise Aspline-degree 3
  
  if (degree==3){
    x_points = c(min(x),final_knots,max(x))
    intercepts = rep(0, length(final_knots)+1)
    first = rep(0, length(final_knots)+1)
    second = rep(0, length(final_knots)+1)
    third = rep(0, length(final_knots)+1)
    
    for (i in 1:length(intercepts)) {
      df <- subset(a_predict, x >= x_points[i] & x <= x_points[i+1])
      tempmodel_coefs <- coef(lm(pred ~ x + I(x^2) + I(x^3),data=df))
      intercepts[i] = tempmodel_coefs[1]
      first[i] = tempmodel_coefs[2]
      second[i] = tempmodel_coefs[3]
      third[i] = tempmodel_coefs[4]
    }
    
    areas = rep(0, length(final_knots)+1)
    
    for (i in 1:length(areas)) {
      func = function(x,k=i) {return(third[k]*x^3 + second[k]*x^2 + first[k]*x + intercepts[k])}
      areas[i] = integral(func, x_points[i], x_points[i+1])
    }
    
    relative_proportions <- rep(0, length(final_knots)+1)
    
    for (i in 1:length(relative_proportions)) {
      relative_proportions[i] = areas[i]/sum(areas)
    }
    
    return_list <- list(relative_proportions,final_knots,plot,intercepts,first,second,third,sum(areas),a_predict)
  }
  return(return_list)
}


# Loess_Area_Proportions function

loess_wrapper_extrapolate <- function(x, y, span.vals = seq(0.25, 1, by = 0.05), folds = 5){
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
  return(best.model$fitted)
}



Loess_Area_Proportions_trapz <- function(fitted,knot_list){
  x_points = c(min(x),knot_list,max(x))
  data = data.frame(x,fitted)
  
  areas = rep(0, length(knot_list)+1)
  
  for (i in 1:length(areas)){
    df <- subset(data, x >= x_points[i] & x <= x_points[i+1])
    areas[i] <- trapz(df$x,df$fitted)
  }
  
  relative_proportions <- rep(0, length(knot_list)+1)
  
  for (i in 1:length(relative_proportions)) {
    relative_proportions[i] = areas[i]/sum(areas)
  }
  
  plot <- ggplot() +
    geom_point(data = data, aes(x, y), shape = 1) +
    geom_line(data = data, aes(x, fitted, color='red'), size = 0.9) +
    geom_vline(xintercept = knot_list, color = 'blue',linetype="dotted", size = 0.75) +
    scale_color_hue(labels = c("Loess Estimates Fit")) + 
    theme(legend.position = "top") +
    guides(color=guide_legend("")) +
    ylab("Velocity") +
    xlab("Time of the day (in seconds)")
  
  return(list(relative_proportions,knot_list,plot))
}

Loess_Area_Proportions_simpsons <- function(fitted,knot_list){
  x_points = c(min(x),knot_list,max(x))
  data = data.frame(x,fitted)
  
  areas = rep(0, length(knot_list)+1)
  
  for (i in 1:length(areas)){
    df <- subset(data, x >= x_points[i] & x <= x_points[i+1])
    areas[i] <- sintegral(df$x,df$fitted)$int
  }
  
  relative_proportions <- rep(0, length(knot_list)+1)
  
  for (i in 1:length(relative_proportions)) {
    relative_proportions[i] = areas[i]/sum(areas)
  }
  
  plot <- ggplot() +
    geom_point(data = data, aes(x, y), shape = 1) +
    geom_line(data = data, aes(x, fitted, color='red'), size = 0.9) +
    geom_vline(xintercept = knot_list, color = 'blue',linetype="dotted", size = 0.75) +
    scale_color_hue(labels = c("Loess Estimates Fit")) + 
    theme(legend.position = "top") +
    guides(color=guide_legend("")) +
    ylab("Velocity") +
    xlab("Time of the day (in seconds)")
  
  return(list(relative_proportions,knot_list,plot))
}


# For Demo on Wednesday

#Loess_Area_Proportions_trapz(fitted,Aspline_Area_Proportions(1)[[2]])

#Loess_Area_Proportions_simpsons(fitted,Aspline_Area_Proportions(1)[[2]])

#Aspline_Area_Proportions(1)

#Loess_Area_Proportions_trapz(fitted,Aspline_Area_Proportions(3)[[2]])

#Loess_Area_Proportions_simpsons(fitted,Aspline_Area_Proportions(3)[[2]])

#Aspline_Area_Proportions(3)

# Getting the normalized full function form (PDF) with help of indicators.

Normalized_fullfunction_Aspline1 <- function(t){
  
  list = Aspline_Area_Proportions(1)
  knot_list = list[[2]]
  intercepts = list[[4]]
  slopes = list[[5]]
  
  x_points = c(min(x),knot_list,max(x))
  
  func_value = 0
  for (i in 1:(length(knot_list)+1)){
    func_value = func_value + (slopes[i]*t + intercepts[i])*as.numeric(I(t >= x_points[i] & t <= x_points[i+1]))
  }
  
  return((1/Aspline_Area_Proportions(1)[[6]])*func_value)
}


Normalized_fullfunction_Aspline3 <- function(t){
  
  list <- Aspline_Area_Proportions(3)
  knot_list = list[[2]]
  intercepts = list[[4]]
  first = list[[5]]
  second = list[[6]]
  third = list[[7]]
  
  x_points = c(min(x),knot_list,max(x))
  
  func_value = 0
  for (i in 1:(length(knot_list)+1)){
    func_value = func_value + (third[i]*t^3 + second[i]*t^2 + first[i]*t + intercepts[i])*as.numeric(I(t >= x_points[i] & t <= x_points[i+1]))
  }
  
  return((1/Aspline_Area_Proportions(3)[[8]])*func_value)
}

Normalized_piecewise_Aspline1 <- function(t,k){
  
  list = Aspline_Area_Proportions(1)
  intercepts = list[[4]]
  slopes = list[[5]]
  
  return((1/Aspline_Area_Proportions(1)[[6]])*(slopes[k]*t + intercepts[k]))
}

Normalized_piecewise_Aspline3 <- function(t,k){
  
  list = Aspline_Area_Proportions(3)
  intercepts = list[[4]]
  first = list[[5]]
  second = list[[6]]
  third = list[[7]]
  
  return((1/Aspline_Area_Proportions(3)[[8]])*(third[k]*t^3 + second[k]*t^2 + first[k]*t + intercepts[k]))
}

# Getting CDF's equations setup

CDF_Aspline1 <- function(t){
  return(integral(Normalized_fullfunction_Aspline1, min(x), t))
}

CDF_Aspline3 <- function(t){
  return(integral(Normalized_fullfunction_Aspline3, min(x), t))
}

Inverse_CDF_Aspline1 <- function(r){
  return(uniroot(function(t){return(integral(Normalized_fullfunction_Aspline1, min(x), t)-r)},lower = min(x), upper = max(x), tol = 1e-9)$root)
}

Inverse_CDF_Aspline3 <- function(r){
  return(uniroot(function(t){return(integral(Normalized_fullfunction_Aspline3, min(x), t)-r)}, lower = min(x), upper = max(x), tol = 1e-9)$root)
}

# Improvement to increase the computation speed.

Speedy_Inverse_CDF_Aspline1 <- function(t){
  
  list = Aspline_Area_Proportions(1)
  area_props = list[[1]]
  knot_list = list[[2]]
  intercepts = list[[4]]
  slopes = list[[5]]
  
  cumul_area_props <- append(cumsum(area_props), 0, after = 0)
  for (i in 1:length(cumul_area_props)){
    if (cumul_area_props[i] <= t & cumul_area_props[i+1] >= t){
      root = uniroot(function(t,k=i){return(integral(Normalized_piecewise_Aspline1, min(x), t)-r)}, lower = min(x), upper = max(x), tol = 1e-9)$root
      
    }
  }
}

