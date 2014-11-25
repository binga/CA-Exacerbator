#### Exacerbation Prediciton ####
# 1. Read the dataset. Look at the snapshot such as basic statistics
# 2. Convert the data types of each variables
# 4. Adopt a missing value strategy
# 5. 2 sets of features available. Medical and genetic. Frame strategy
# 6. Sample into train, test and evaluation. Stratified sampling.
# 7. Run models and finally do cross-validation
# 8. 

#### 1. Reading data ####
setwd("D:/CA/Exacerbation/")
data <- read.csv("CAX_ExacerbationModeling_TRAIN_data.csv")
metadata <- read.csv("CAX_ExacerbationModeling_MetaData.csv")
test <- read.csv("CAX_ExacerbationModeling_Public_TEST_data.csv")
data.total <- rbind(data, test)

#### 2. Changing format of dataset ####
for(i in 1:nrow(metadata)){
  if(metadata$Column.Type[i] =="Numeric"){
    data.total[,i] <- as.numeric(as.character(data.total[,i]))
  }else if(metadata$Column.Type[i] == "Category"){
    data.total[,i] <- as.factor(as.character(data.total[,i]))
  }else if(metadata$Column.Type[i] == "Ordered Category"){
    data.total[,i] <- factor(data.total[,i], ordered = T)
  }
}
rm(i)
# Sanity check
table(unlist(lapply(data.total, function(x){class(x)})))

#### 3. Impute the data using a missing value strategy ####
library(DMwR)
data.total.imputed <- centralImputation(data.total)

#### 4. Split the entire data into its original forms ####
train <- data.total.imputed[1:4099,]
eval <- data.total.imputed[4100:nrow(data.total.imputed), ]
rm(data, data.total, test, data.total, data.total.imputed)

#### 5. Sampling ####
library(sampling)
train <- train[order(train$Exacebator), ]
samples <- strata(train, stratanames=c("Exacebator"), size=c(343, 343),
                  method="srswor")
tr <- getdata(train, samples)
te <- train[-tr$ID_unit, ]
tr <- tr[,-c(1333:1335)]
tr <- data.frame("sid"=tr[,1], "Exacebator"=tr$Exacebator, tr[,c(2:1331)])
rm(samples)

## Scaling
match("numeric", lapply(tr, function(x){class(x)}))
indices <- as.character(metadata[metadata$Column.Type=="Numeric","varnum"])
tr[ ,indices] <- scale(tr[ ,indices], center = T, scale = T)

#### 6. Modeling ####
library(caret)
library(LiblineaR)
C <- heuristicC(data.matrix(tr[,-c(1)]))
libRmod <- LiblineaR(tr[,-c(1)], tr$Exacebator, type = 5, cost = C,
                     cross = 10)
libRmod <- LiblineaR(tr[,-c(1)], tr$Exacebator, type = 5, cost = C)

str(tr)


## Random Forest
library(randomForest)
tuneRF(tr[,-c(1)], tr$Exacebator,stepFactor =  )
rfmod <- randomForest(Exacebator ~. , data = tr[,-c(1)], ntree = 100)
te.pred.rf <- predict(rfmod, te[,-c(1)])


#### 7. Test set submission ####
submissionTemplate <- read.csv("CAX_ExacerbationModeling_SubmissionTemplate.csv")
submissionTemplate[,2] <- predict(rfmod, eval[,-c(1)])
table(submissionTemplate$Exacebator)

write.csv(submissionTemplate, "SS_50_50_RF_Scaling.csv", row.names = F)
getwd()

write.csv(tr, "Pypy/Train_Scaled.csv", row.names = F)
