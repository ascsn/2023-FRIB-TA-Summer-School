# example.R
# Simple example showing fitting various models to simple test datasets,
# including saving fit to disk and re-loading to run predictions or 
# extract variable activity information.


#-----------------------------------------------------------------------
# Test Branin function, rescaled
# see: http://www.sfu.ca/~ssurjano/
#-----------------------------------------------------------------------
braninsc <- function(xx)
{  
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15*x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - 5.1*x1bar^2/(4*pi^2) + 5*x1bar/pi - 6
  term2 <- (10 - 10/(8*pi)) * cos(x1bar)
  
  y <- (term1^2 + term2 - 44.81) / 51.95
  return(y)
}


#-----------------------------------------------------------------------
# Simulate branin data for testing
#-----------------------------------------------------------------------
set.seed(99)
n=500
p=2
x = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = braninsc(x[i,])

# Load the R wrapper functions to the OpenBT library.
source("Documents/Open BT Project SRC/openbt.R")

# Homoscedasitc BART model
fit=openbt(x,y,pbd=c(0.7,0.0),ntreeh=1,numcut=100,tc=4,model="bart",modelname="branin")

# Heteroscedastic HBART model
# fit=openbt(x,y,k=10,numcut=100,tc=4,model="hbart",modelname="branin")

# Truncated BART model (c/o Dai Feng of Merck)
# tv=sample(1:n,5)
# miny=min(y)
# y[tv]=miny
# fit=openbt(x,y,pbd=c(0.7,0.0),ntreeh=1,numcut=100,tc=4,model="merck_truncated",modelname="branin")


# Extract posterior tree samples in vector coding
trees=openbt.scanpost(fit)
trees$mt[[1]][[1]]
trees$st[[1]][[1]]


# Save fitted model to local directory
openbt.save(fit,"test")


# Load fitted model to a new object.
fit2=openbt.load("test")

# Note that saved models are saved in compressed format and so 
# are not human readable.  The loaded models are uncompressed
# into a temporary working directory.  These directories
# are automatically deleted upon exiting R.
# Generally the user does not need to care about these
# behind-the-scenes details.
fit$folder
fit2$folder


# Predict the underlying response function
fitp=predict.openbt(fit2,x,tc=4)

# Calculate variable activity information
fitv=vartivity.openbt(fit2)

# Plot fitted model
plot(y,fitp$mmean,xlab="observed",ylab="fitted")
abline(0,1)

# 3d plot if you like
library(rgl)
plot3d(x[,1],x[,2],y)
points3d(x[,1],x[,2],fitp$mmean,col="red")
points3d(x[,1],x[,2],fitp$mmean+2*fitp$smean,col="pink")
points3d(x[,1],x[,2],fitp$mmean-2*fitp$smean,col="pink")

# Plot variable activity
plot(fitv)

# Calculate Sobol indices
fits=sobol.openbt(fit2)
fits$msi
fits$mtsi
fits$msij

# Example of MO optimization
fit.b=openbt(x,y,pbd=c(0.7,0.0),ntreeh=1,numcut=100,tc=4,model="bart",modelname="branin-b")

pf=mopareto.openbt(fit,fit.b,tc=24) # quite slow, use many cores to speed it up

# plot front
par(mfrow=c(1,2))
plot(0,0,xlim=c(-2,2),ylim=c(-2,2),type="n",xlab="y1",ylab="y2")
for(i in length(pf)) points(t(pf[[i]]$theta),pch=20)
# plot set
plot(0,0,xlim=c(0,1),ylim=c(0,1),type="n",xlab="x1",ylab="x2")
for(i in 1:length(pf)) rect(pf[[i]]$a[1,],pf[[i]]$a[2,],pf[[i]]$b[1,],pf[[i]]$b[2,],col=gray(0.5,alpha=0.1),border=NA)

