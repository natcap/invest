
###
###  Computation (and some data pre-processing) for recreation model
###
###     sessid is passed as an argument to this script using '--args sessid="xxxx"'
###       eg: R CMD BATCH --vanilla '--args a="text" b=c("blue","red", "green") c=3' regression.r
###
###       remaining configuration is done via the server config file
###
###       although additional vars could be pased using, eg, '--args sessid="xxxx" a="text" b=c(2,5,6)'
###
###     to suppress log output, add "/dev/null" to the end of the R CMD BATCH command
###

##
##  Libraries
##

library(foreign)
library(rjson)


##
##  Debugging
##
## start R from /usr/local/recreation/public_html and then
## temporarily set sessid, which is passed to this script via argument
# sessid = "DEBUGGING"
# sessid = "r6umkrdde23glvtb8doamdq137/"


## 
##  User-defined
##

# options(width=Sys.getenv("COLUMNS"))
# su <- summary
# op <- par(no.readonly=TRUE)
# .libPaths()


##
##  Get arguments
##

## First read in the arguments listed at the command line
## ## args becomes a list of character vectors
args=(commandArgs(TRUE))

## Check for arguments
## ## if no arguments, break
if(length(args)==0) {
  print("No arguments supplied.")
#   break
  ## would be possible to supply default values...
#     a = 1
## ## else assign arguments to object
} else {
  for(i in 1:length(args)){
        eval(parse(text=args[[i]]))
  }
}

## Then get additional arguments from the config file
pathconf <- "./recreation_server_config.json"
conf <- fromJSON(paste(readLines(pathconf), collapse=""))
## ## store path names
userpath <- paste(conf$paths$absolute$userroot, conf$paths$relative$data, sep='')
webpath <- paste(conf$paths$absolute$webroot, conf$paths$relative$data, sep='')
sesspath <- paste(userpath, sessid, "/", sep='')
recpath <- conf$paths$absolute$repository
compath <- conf$paths$relative$computation
## ## store file names
gridname <- conf$files$grid$dbf
gridpubnam <- conf$files$grid_public$dbf
excludecol <- conf$postgis$table$restrict
flickrname <- conf$files$flickr
paramname <- conf$files$params
logname <- conf$files$log

## determine whether this is an initial vs scenario run
## ## using the presence of the scenario.json file
if (file.exists(paste(sesspath,"scenario.json", sep=''))) {
  scenario <- "y"
} else {
  scenario <- "n"
}



##
##  Logging function
##

## send a message to the log file
## ## type = INFO, DEBUG, WARNING, or ERROR
## ## msg = any text, no commas, no period at the end
## ## sp = sessionpath
## ## ln = logname
tolog <- function(typ, msg, sp=sesspath, ln=logname) {
  cat(paste(system("date +'%m/%d/%Y %H:%M:%S'", intern=T),",",typ,",",msg,".",sep=''), file=paste(sesspath,logname,sep=''), sep="\n", append=T)
  ## fyi, am calling the shell date command because any fxn using R's datetime class crashes R on tenas
  ## ## frustrating!  but this works fine.
}


##
##  Import data
##

## Grid
grid <- read.dbf(paste(sesspath, gridname, sep=''))

## Flickr photos
fa.phs <- read.csv(paste(sesspath, flickrname, sep=''))

tolog("INFO","Read data for regression computation")

## Parameter estimates (if scenario run)
if (scenario=="y") {
  scepar <- read.csv(paste(sesspath, paramname, sep=''))
}


##
##  Photo-user-days 
## 

## Calculate number of photos per user per grid cell
## ## this fxn has been moved to recreation.py

## Calculate total user days per cell
## ## For now, use 2005 - 2010 data
sd <- "2005-01-01"
ed <- "2012-12-31"
## ## calculate the number of months
sdd <- as.Date(sd, "%Y-%m-%d")
edd <- as.Date(ed, "%Y-%m-%d")
mo <- round(as.numeric((edd-sdd)/(365/12)))
tolog("INFO","Calculations use mean annual photo-user-days from 2005-2012")
## ## subset the data
forusds <- fa.phs[as.character(fa.phs$date_taken) >= sd & as.character(fa.phs$date_taken) <= ed, ]
fa.usds <- aggregate(list(usds=forusds$piclen), list(cellID=forusds$cellID), length)
## ## total user days per MONTH per cell (note: cell-mos with no photos are missing here!)
fa.usmo <- aggregate(list(usdm=forusds$piclen), list(cellID=forusds$cellID, date_taken=substr(forusds$date_taken,1,6)), length)
## ## total user days per YEAR per cell (note: cell-years with no photos are missing here!)
fa.usyr <- aggregate(list(usdy=forusds$piclen), list(cellID=forusds$cellID, date_taken=substr(forusds$date_taken,1,4)), length)
## ## average user days per MONTH per cell (across all mos)
## ## ## note: sum then divide to account for zero cell-mos
# fa.usmo.av <- aggregate(list(usdmav=fa.usmo$usdm), list(cellID=fa.usmo$cellID), sum)
# fa.usmo.av$usdmav <- round(fa.usmo.av$usdmav / mo, digits=4)
## ## average user days per YEAR per cell (across all mos) (note: sum then div to account for missing cell-yrs)
fa.usyr.av <- aggregate(list(usdyav=fa.usyr$usdy), list(cellID=fa.usyr$cellID), sum)
fa.usyr.av$usdyav <- round(fa.usyr.av$usdyav / (mo/12), digits=4)
## ## Add proportional user days
# fa.usmo.av$usdmav.pr <- round( (fa.usmo.av$usdmav / sum(fa.usmo.av$usdmav)) * 100 , digits=3)
fa.usyr.av$usdyav.pr <- round( (fa.usyr.av$usdyav / sum(fa.usyr.av$usdyav)) * 100 , digits=3)

tolog("INFO","Calculated photo-user-days")


##
##  QA/QC check for sufficiently large AOI
##

## Warn the user if more than 40% of the cells contain no photo-user-days
if ((nrow(fa.usyr.av) / nrow(grid)) < .40) {
  pcellswdata <- round(100 * (nrow(fa.usyr.av) / nrow(grid)), 1)
  tolog("WARNING",paste(pcellswdata,"% of cells contain photos", sep=''))
  tolog("WARNING","We suggest increasing the cell size to estimate effects of the predictors")
}


##
##  Data assembly and clean-up
##

## Merge predictors with photos
fortop <- merge(fa.usyr.av[ ,c("cellID","usdyav","usdyav.pr")], grid, by="cellID", all=T)

## Get list of predictor names and their column positions
# fortop <- grid@data
prepos <- which(colnames(fortop) != "cellID" & colnames(fortop) != "cellArea" & colnames(fortop) != "usdyav" & colnames(fortop) != "usdyav.pr")
prenam <- colnames(fortop)[prepos]

## Replace NA's with zeros
for (i in 1:ncol(fortop)) {
  fortop[is.na(fortop[ ,i]) , i] <- 0
}



##
## Parameterization (and some plotting)
##

top <- fortop

## Potentially drop points with extreme leverage
# top <- top[-c(4495), ]

## plot histogram
# histogram(log(top$usdyav.pr+1), xlab="cell monthly visitation (log user-days)")

## plot each response X dependent var
# par(op)
# par(mfrow=c(3,3))
#   plot(log(top$usdyav.pr+1) ~ (top$percland+1))
#   plot(log(top$usdyav.pr+1) ~ log(top$airps+1))
#   ## etc...
# par(op)


## linear model
## ## log-tranformed response variable
vv <- lm( paste("log1p(top$usdyav) ~", paste(" top$",prenam[1:length(prenam)],collapse=" + ",sep='')) )
# vv <- lm( paste("log1p(top$usdyav) ~", paste(" log1p(top$",prenam[1:length(prenam)],collapse=") + ",sep=''), ")", sep="") )
summary(vv)
## ## export summary graph
pdf( paste(sesspath, "regression_summary.pdf", sep=''), width=7, height=7)
par(mfrow=c(2,2)); plot(vv); # par(op);
dev.off()

tolog("INFO","Performed linear regression")


##
##  Estimation
##

## get the regression coefs
# vvcoefs <- lapply(c(as.vector(coef(vv))), function(x) x)
if (scenario=="y") {
  scepar[,1] <- tolower(scepar[,1])
  vvcoefs <- NA
  vvcoefs[1] <- scepar$Estimate[1]
  for (i in 2:length(scepar$Estimate)){
    vvcoefs[i] <- scepar$Estimate[which(scepar[,1] == prenam[i-1])]
  }
  # vvcoefs <- scepar$Estimate
} else {
  vvcoefs <- as.vector(coef(vv))
}
## make the predictions
forest <- fortop
## ## only if the model succeeded to estimate a coef for all predictors
if (scenario == "n" & NA %in% coef(vv)) {
  ## ## warn user if one or more coefficients could not be estimated
  tolog("WARNING","Effect size of one or more predictors could not be estimated")
  tolog("WARNING","Without all effect sizes visitation cannot be estimated")
  forest$usdyav.est <- rep(NA, nrow(forest))
} else {
  ## ## for log-tranformed response var
  vvexp <- parse(text=paste("expm1(with(forest, vvcoefs[1] +", paste0("vvcoefs[",(1:length(prenam))+1,"] * ",prenam[1:length(prenam)],"", collapse=" + "), "))"))
  forest$usdyav.est <- eval(vvexp)
}



##
##  EXPORT
##

##  export estimation data as a dbf for a shapefile
## ## reorder rows to match order of incoming grid
forest <- forest[match(grid$cellID, forest$cellID), ]
## ## don't include sensitive data that we shouldn't redistribute
write.dbf(forest[ ,!(colnames(forest) %in% excludecol)], paste(sesspath, gridpubnam, sep=''))

## export the regression stats as a csv (if init run)
if (scenario=="n") {
  params <- signif(summary(vv)$coefficients, 4)
  ## ## bind on coefficients which were NA (couldn't be estimated)
  if (NA %in% coef(vv)) {
    ## ## create a NA matrix with rows for NA coefs
    parnas <- names(coef(vv)[!(names(coef(vv)) %in% rownames(params))])
    parvas <- matrix(rep(NaN, length(parnas)*4), ncol=4)
    rownames(parvas) <- parnas
    ## ## merge onto matrix of estimated coefs
    params <- rbind(params , parvas)
    ## ## reorder
    params <- params[match(names(coef(vv)) , rownames(params)), ]
  }
  ## ## write the file
  row.names(params) <- c("Intercept", prenam)
  write.csv(params, paste(sesspath, paramname, sep=''))

}

tolog("INFO","Wrote regression statistics")
