df <- nfl_cleaned_data
View(df)
df$Conv_per <-(df$`3DConv`/df$`3DAtt`)    #creating 3rd down conversion rate
View(df)
df$Conv_per <- (df$Conv_per)*100 #coverting to a percentage 
View(df)
install.packages("writexl")
library(writexl)
write_xlsx(df, "~/Desktop/my_dataset.xlsx") #saving new excel to laptop
p<-.7
obs_count<-dim(df)[1]
training_size <- floor(p * obs_count)
training_size
set.seed(3721) #setting random seed
train_ind <- sample(obs_count, size = training_size)

Training <- df[train_ind, ] #PULLS RANDOM ROWS FOR TRAINING
Testing <- df[-train_ind, ] #PULLS RANDOM ROWS FOR TESTING

#CHECKING THE DIMENSIONS OF THE PARTITIONED DATA
dim(Training)
dim(Testing)

M1 <- lm(Win ~ OT + Home + PF + PA +`1stDF` + RushYd + PassYd + TOA + `1stDA` + 
           OppRush + OppPass + OppTO + comingoffbye + `3DConv` + `3DAtt`+ Conv_per, Training) #first model
summary(M1) #SUMMARY DIAGNOSTIC OUTPUT


variables <- Training[, c("Win","OT", "Home", "PF", "PA", "1stDF", "RushYd", "PassYd", 
                          "TOA", "1stDA", "OppRush", "OppPass", "OppTO", 
                          "comingoffbye", "3DConv", "3DAtt", "Conv_per")]

# Compute the correlation matrix
cor_matrix <- cor(variables, use = "complete.obs")

# Print the correlation matrix
print(cor_matrix)

PRED_1_IN <- predict(M1, Training) #first model
View(PRED_1_IN) #VIEW IN-SAMPLE PREDICTIONS
View(M1$fitted.values) #FITTED VALUES ARE IN-SAMPLE PREDICTIONS

#GENERATING PREDICTIONS ON THE TEST DATA TO BENCHMARK OUT-OF-SAMPLE PERFORMANCE 
PRED_1_OUT <- predict(M1, Testing) 

#COMPUTING / REPORTING IN-SAMPLE AND OUT-OF-SAMPLE ROOT MEAN SQUARED ERROR
(RMSE_1_IN<-sqrt(sum((PRED_1_IN-Training$Win)^2)/length(PRED_1_IN))) #computes in-sample error
(RMSE_1_OUT<-sqrt(sum((PRED_1_OUT-Testing$Win)^2)/length(PRED_1_OUT))) #computes out-of-sample 



