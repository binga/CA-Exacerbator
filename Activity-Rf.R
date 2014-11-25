#### Exacerbation Prediciton ####
# 1. Read the dataset. Look at the snapshot such as basic statistics
# 2. Identify types of variables using metadata file given
# 3. Convert the data types of each variables
# 4. Adopt a missing value strategy
# 5. 2 sets of features available. Medical and genetic. Frame strategy
# 6. Sample into train, test and evaluation. Stratified sampling.
# 7. Run models and finally do cross-validation
# 8. 


setwd("D:/CA/Exacerbation/")
#setwd("D:/Analytics/CA/Exacerbation/")
data <- read.csv("CAX_ExacerbationModeling_TRAIN_data.csv")
dim(data) # 4099 x 1332
sum(is.na(data)) # 182663 - 3.3%
table(data$Exacebator) # 3756, 343