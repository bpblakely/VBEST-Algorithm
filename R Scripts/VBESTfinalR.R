
#######################################################
### Overall Function for Twitter Sampling VBEST   ####
### Algorithm and other tests                      ###
#######################################################

#### R code written by Ravi Singh and Trent Buskirk and Youzhi Yu
#### Python Implementation Brian Blakely
#### VBEST Version 1
#### October 29, 2020

#### In this experiment we will test three different sampling methods based on our
#### estimated twitter velocities including:
#### 1. Inverse Sampling from the Velocity Curve
#### 2. SImple Random Sample (without replacement) of PSUs created using Estimated Velocity Curve
#### 3. THe VBEST Systematic sample of PSUs created using the ETVC (estimated twitter velocity curve)

#### This function will manage the creation of the requisite samples for each of these
#### aspects of our experiment and should be run once per day/region over the field period
#### to generate the requisite samples and information

#### Function Inputs are described here
#### step1aX = x-vector from stage 1 of the v-best algorithm
#### step2aV = Velocitiy vector corresponding to stage 1 x-values
#### Vbestn = VBEST PSU sample size (we would consider this to be 804)
#### SRSn = SRS of PSU sample size (we would consider this to be 804)
#### IVn = inverse sampling sample size (we would consider this to be 804; the sampling here is done with replacement to simplify estimation process)
#### Please note, SRSn and IVn are now set to be equal to Vbestn. 
#### refine = grid refinement factor to determine the number of points to use for creating estimates of tweet volumes over the entire day (we consider the default to be 1/200=.005)
#### seedvec =vector of three seed values to determine the three different samples

#### required packages needed to run this function

#install.packages("MESS")
#install.packages("tidyverse")
#install.packages("bootstrap")  
#install.packages("itertools")    
#install.packages("foreach")    
#install.packages("doParallel") 
require(MESS)
require(tidyverse)
require(bootstrap)
require(itertools)
require(foreach)
require(doParallel)

loesswrap <- function(x,y, span.vals = seq(0.2, 0.65, by = 0.05), folds = 3){
  
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
  return(list("best.model"=best.model,"best.span"=best.span))
}

VBESTsamp <- function(step1aX, step2aV, seedvec=NULL, Vbestn=624, refine=.005){

#### functions that will be used in this larger function
#loesswrap - which will compute the loess estimate of the Twitter Velocity Curve 
# based on a predetermined span and cross validatiion of 3
#getGrids will use information computed from the loesswrap to create the final samples 
# required from our experiment

#### Function will Returned a collection of Grids for Twitter Sampling as well as additional
#### metatdata about our model and additional information needed for weighting and 
#### adjustment at the estimation phase including: 
#### 1 - VBEST Sample of PSUs (corresponding to a grid of time points in descending order to be used to query twitter)
#### 2 - SRS Sample of PSUs (corresponding to a grid of time points in descending order to be used to query twitter)
#### 3 - INV Sample (corresponding to a grid of points in descending order...)
#### 4 - Span Value from Estimated Tweet Velocity Curve Model
#### 5 - Total Tweet Volume Estimated
#### 6 - Total Number of TweetPSUs created by VBEST algorithm
#### 7 - Estimated Expected Tweets per PSU grid (length of this vector equals value in 6)
#### 8 - Overage of Expected Tweets in each TweetPSU (returned value in 7 - 100)
#### 9 - Sampling Weights to be used for inverse sampling estimates


#### VBEST Step 1 is performed outside of this R function

#### VBEST Step 2a - estimation of twitter velocity from initial sample is also 
#### performed outside of this function but these are key inputs to this function

### VBEST Step 2b (estimate Tweet Velocity Curve)
  
if (is.null(seedvec)){
  seedvec <- c(sample(1:100000, 10, replace=FALSE))
}

set.seed(seedvec[10])
tweetvelcurv<-loesswrap(step1aX, step2aV, span.vals=seq(.2,.65, by=.05), folds=3)
model<-tweetvelcurv$best.model

### VBEST Step 3 (Create TweetPSUs)
## First we will create expanded grid of x-points to estimate total area under twitter velocity curve
x_grid <- round(seq(0,86400,refine),5)

cores_used <- ceiling(detectCores()/2)
cl <- makeCluster(cores_used)
registerDoParallel(cl)

y_hats <-
  round(foreach(d=isplitVector(x_grid, chunks = cores_used),
                .combine=c, .packages=c("stats")) %dopar% {
                  predict(model, newdata=d)
                },5)

stopCluster(cl)

# Implementing trapz using 3 consecutive points. Using the matrix multiplication.

coeffs <- matrix(c(refine/2,refine,refine/2), nrow = 1, byrow = TRUE)

row1 <- y_hats[seq(1,length(x_grid)-2,2)]
row2 <- y_hats[seq(2,length(x_grid)-1,2)]
row3 <- y_hats[seq(3,length(x_grid),2)]

m <- matrix(c(row1,row2,row3), nrow = 3, byrow = TRUE)

Loessareas <- round(as.vector(coeffs%*%m),12)

timeseq  <- round(seq(2*refine, 86400, length.out = 86400/(2*refine)),5)

###### now we will construct the TweetPSUS
tempdf<-data.frame(rev(timeseq),rev(Loessareas))
names(tempdf)<-c("PSUPoint","EstVol")
tempdf<-tempdf %>%
  group_by(group_100 = cumsumbinning(EstVol,99.9999999999999999, cutwhenpassed =TRUE)) %>%
  mutate(cumsum_100 = cumsum(EstVol)) 

tempdf2<-tempdf %>%
  group_by(group_100) %>%
  filter(row_number() == n())

finaldf<-tempdf %>%
  group_by(group_100) %>%
  filter(row_number() == 1)

finaldf$cumsum_100<-tempdf2$cumsum_100
index = nrow(finaldf)
if (finaldf$cumsum_100[index] < 50){
  last_cumsum_100 <- finaldf$cumsum_100[index]
  finaldf <- head(finaldf,-1) # This removes the last row.
  finaldf$cumsum_100[index-1] = finaldf$cumsum_100[index-1] + last_cumsum_100
}

numPSUs<-dim(finaldf)[1]
PSUGrid<-finaldf$PSUPoint

# samplesize <- Vbestn

vbestList <- vector(mode = "list",length=3)
srsList <- vector(mode = "list",length=3)
invList <- vector(mode = "list",length=3)
weightList <- vector(mode = "list",length=3)

for (i in 1:3){
  
  samplesize <- Vbestn - 180*(i-1)
  
  if (numPSUs < samplesize){
    samplesize <- numPSUs
  }
  sampint<-numPSUs/samplesize
  samp<-NULL
  set.seed(seedvec[1*i])
  randintX<-sample(1:numPSUs,1)
  samp[1]<-randintX
  
  for (j in 1:(samplesize-1)){
    samp[j+1]<-(round((randintX+j*sampint),0))%%numPSUs
  }
  
  samp<-sort(replace(samp,which(samp==0),numPSUs))
  vbestList[[i]]<-PSUGrid[samp]
  
  ### Second Step is SRS sample of TweetPSUs
  samp<-NULL
  set.seed(seedvec[2*i])
  samp<-sample(1:numPSUs,samplesize, replace=FALSE)
  samp<-(sort(samp))
  srsList[[i]]<-PSUGrid[samp]
  
  ### Third Step is the inv sample step
  weightvec<-Loessareas/sum(Loessareas)
  set.seed(seedvec[3*i])
  samp<-NULL
  samp<-sample(1:length(timeseq),size=samplesize, replace=TRUE, prob=weightvec)
  samporder<-order(samp, decreasing=TRUE)
  invList[[i]]<-timeseq[samp][samporder]
  weightList[[i]]<-1/weightvec[samp][samporder]
}

return(list("vbestpsus"=vbestList,"srspsus"=srsList,"invsamp"=invList,"spanval"=tweetvelcurv$best.span,"totaltweets"=sum(Loessareas),"NPSUs"=numPSUs,"exptweets"=finaldf$cumsum_100, "overtweets" =finaldf$cumsum_100-100 , "invsampwts"=weightList, "samplesize"=Vbestn, "seeds"=seedvec))
}




VBESTshort<- function(step1aX, step2aV, seedvec=NULL, Vbestn=624, refine=.005){

if (is.null(seedvec)){
  seedvec <- c(sample(1:100000, 10, replace=FALSE))
}

set.seed(seedvec[10])
tweetvelcurv<-loesswrap(step1aX, step2aV, span.vals=seq(.2,.65, by=.05), folds=3)
model<-tweetvelcurv$best.model

### VBEST Step 3 (Create TweetPSUs)
## First we will create expanded grid of x-points to estimate total area under twitter velocity curve
x_grid <- round(seq(0,86400,refine),5)

cores_used <- ceiling(detectCores()/2)
cl <- makeCluster(cores_used)
registerDoParallel(cl)

y_hats <-
  round(foreach(d=isplitVector(x_grid, chunks = cores_used),
                .combine=c, .packages=c("stats")) %dopar% {
                  predict(model, newdata=d)
                },5)

stopCluster(cl)

# Implementing trapz using 3 consecutive points. Using the matrix multiplication.

coeffs <- matrix(c(refine/2,refine,refine/2), nrow = 1, byrow = TRUE)

row1 <- y_hats[seq(1,length(x_grid)-2,2)]
row2 <- y_hats[seq(2,length(x_grid)-1,2)]
row3 <- y_hats[seq(3,length(x_grid),2)]

m <- matrix(c(row1,row2,row3), nrow = 3, byrow = TRUE)

Loessareas <- round(as.vector(coeffs%*%m),12)

Loess_x_grid2 = seq(0,86399,1)
y_hats2 <- predict(model,Loess_x_grid2)

return(list("y_hats" = y_hats2, "spanval"=tweetvelcurv$best.span,"totaltweets"=sum(Loessareas),"seed"=seedvec[10]))
}


