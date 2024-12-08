df <- NFl_big_data_cleaned
View(df)
df$Conv_per <-(df$`3DConv`/df$`3DAtt`)    #creating 3rd down conversion rate
View(df)
df$Conv_per <- (df$Conv_per)*100 #coverting to a percentage 
View(df)
library(caret) #used for confusionMatrix()
#some exploratory analysis
df <- NFl_big_data_cleaned
head(df)
summary(df)
dim(df)
View(df)
LPM_0 <- lm(Win~., data=df)
summary(LPM_0$fitted.values)
summary(LPM_0)
# Create dummy variables for specific teams
df$OppPittsburghSteelers <- ifelse(df$Opp == "Pittsburgh Steelers", 1, 0)
df$OppKansasCityChiefs <- ifelse(df$Opp == "Kansas City Chiefs", 1, 0)
df$TeamPittsburghSteelers <- ifelse(df$Team == "Pittsburgh Steelers", 1, 0)
df$TeamKansasCityChiefs <- ifelse(df$Team == "Kansas City Chiefs", 1, 0)

install.packages("writexl")
library(writexl)
write_xlsx(df, "~/Desktop/my_dataset.xlsx")
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
names(Training)
M0 <- lm(Win ~  OppTO , Training) #first bivariate model

M1 <- lm(Win ~ OT + Home + PF + PA + `1stDF` + RushYd + PassYd + TOA + `1stDA` + 
           OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
           `3DConv` + `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
         data = Training) # first multivariate model
summary(M1) #SUMMARY DIAGNOSTIC OUTPUT

variables1 <- Training[, c("Win","OT", "Home", "PF", "PA", "1stDF", "RushYd", "PassYd", 
                          "TOA", "1stDA", "OppRush", "OppPass", "OppTO", "comingoffbye", 
                          "3DConv", "3DAtt", "Conv_per", "TeamKansasCityChiefs", 
                          "OppKansasCityChiefs","TeamPittsburghSteelers", "OppPittsburghSteelers" )]

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



M3 <- lm(Win ~ OT + Home + PF + PA + `1stDF` + RushYd + PassYd + TOA + `1stDA` + 
           OppRush + OppPass + OppTO + comingoffbye + TeamKansasCityChiefs +
           `3DConv` + `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
         data = Training)
summary(M3) #SUMMARY DIAGNOSTIC OUTPUT
PRED_3_IN <- predict(M3, Training) #first model
View(PRED_3_IN) #VIEW IN-SAMPLE PREDICTIONS
View(M3$fitted.values) #FITTED VALUES ARE IN-SAMPLE PREDICTIONS

#GENERATING PREDICTIONS ON THE TEST DATA TO BENCHMARK OUT-OF-SAMPLE PERFORMANCE 
PRED_3_OUT <- predict(M3, Testing) 

#COMPUTING / REPORTING IN-SAMPLE AND OUT-OF-SAMPLE ROOT MEAN SQUARED ERROR
(RMSE_3_IN<-sqrt(sum((PRED_3_IN-Training$Win)^2)/length(PRED_3_IN))) #computes in-sample error
(RMSE_3_OUT<-sqrt(sum((PRED_3_OUT-Testing$Win)^2)/length(PRED_3_OUT))) #computes out-of-sample 

variables <- Training[, c("Win","OT", "Home", "PF", "PA", "1stDF", "RushYd", "PassYd", 
                          "TOA", "1stDA", "OppRush", "OppPass", "OppTO", "comingoffbye", 
                          "3DConv", "3DAtt", "Conv_per", "TeamKansasCityChiefs", 
                          "OppKansasCityChiefs","TeamPittsburghSteelers" )]

# Compute the correlation matrix
cor_matrix <- cor(variables, use = "complete.obs")

# Print the correlation matrix
print(cor_matrix)

# building Logit model

#Setting seed and partitioning data 
p<-.7

#number of observations (rows) in the dataframe
obs_count<-dim(df)[1]

#number of observations to be selected for the training partition
#the floor() function rounds down to the nearest integer
training_size <- floor(p * obs_count)
training_size
#set the seed to make your partition reproducible
set.seed(3721)
#create a vector with the shuffled row numbers of the original dataset
train_ind <- sample(obs_count, size = training_size)

Training <- df[train_ind, ] #pulls random rows for training
Testing <- df[-train_ind, ] #pulls random rows for testing

dim(Training)
dim(Testing)
#CHECKING THE DIMENSIONS OF THE PARTITIONED DATA
dim(Training)
dim(Testing)
names(Training)

#logit model1
M_LOG<-glm(Win ~ OT + Home + PF + PA + `1stDF` + RushYd + PassYd + TOA + `1stDA` + 
             OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
             `3DConv` + `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
           data = Training, family = binomial(link="logit"))
summary(M_LOG)
install.packages("car")  # Install the car package
library(car)   
vif(M_LOG) # testing VIF to Identify multicolinearity
variables <- Training[, c("Win","OT", "Home", "PF", "PA", "1stDF", "RushYd", "PassYd", 
                          "TOA", "1stDA", "OppRush", "OppPass", "OppTO", "comingoffbye", 
                          "3DConv", "3DAtt", "Conv_per", "TeamKansasCityChiefs", 
                          "OppKansasCityChiefs","TeamPittsburghSteelers" )]

# Compute the correlation matrix
cor_matrix <- cor(variables, use = "complete.obs")

# Print the correlation matrix
print(cor_matrix)

#found Multicollinearity so we will drop variables correlated variables causing most issues and run VIF again  
M_LOG2<-glm(Win ~ OT + Home + PF + PA + RushYd + PassYd + TOA + 
             OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
             `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
           data = Training, family = binomial(link="logit"))
summary(M_LOG2)
vif(M_LOG2)
variables <- Training[, c("Win","OT", "Home", "PF", "PA", "RushYd", "PassYd", 
                          "TOA", "OppRush", "OppPass", "OppTO", "comingoffbye", 
                         "3DAtt", "Conv_per", "TeamKansasCityChiefs", 
                          "OppKansasCityChiefs","TeamPittsburghSteelers" )]

# Compute the correlation matrix
cor_matrix <- cor(variables, use = "complete.obs")

# Print the correlation matrix
print(cor_matrix)

# running 2 logit regression to see which produces Lower AIC and residual deviance 
M_LOG3<-glm(Win ~ OT + Home + PF + RushYd + PassYd + TOA + 
              OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
              `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
            data = Training, family = binomial(link="logit"))
summary(M_LOG3)
vif(M_LOG3)

M_LOG4<-glm(Win ~ OT + Home + PA + RushYd + PassYd + TOA + 
              OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
              `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
            data = Training, family = binomial(link="logit"))
summary(M_LOG4)
vif(M_LOG4)
# we will proceed with model 4 but test both against the ROC Curve and AUC
#takes the coefficients to the base e for odds-ratio interpretation
exp(cbind(M_LOG4$coefficients, confint(M_LOG4)))

#generating predicted probabilities
predictions<-predict(M_LOG4, Training, type="response")

#converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
binpredict <- (predictions >= .35)
View(binpredict)

#build confusion matrix based on binary prediction in-sample
confusion<-table(binpredict, Training$Win == 1)
confusion

#summary analysis of confusion matrix in-sample
confusionMatrix(confusion, positive='TRUE') 

#builds the confusion matrix to look at accuracy on testing data out-of-sample
confusionMatrix(table(predict(M_LOG4, Testing, type="response") >= 0.5,
                      Testing$Win == 1), positive = 'TRUE')
#probit model


#ROC & AUC ANALYSIS

library(tidymodels) #FOR YARDSTICK PACKAGE
library(pROC)
roc_obj <- roc(pva$actual, pva$preds)

#NOTE THIS PLOTS SENSITIVITY (TRUE POSITIVES) VS. SPECIFICITY (TRUE NEGATIVES)
plot(roc_obj, col='blue', main="ROC Curve")

(auc <- 1-auc(roc_obj)) #SINCE PICTURE IS INVERTED

#Starting SVM
library(rsample) #FOR initial_split() STRATIFIED RANDOM SAMPLING
library(e1071) #SVM LIBRARY
df$Win<-as.factor(df$Win) #FOR tune.svm()

#CREATING A BALANCED FOLD BY STRATIFYING ON THE OUTCOME
set.seed(3721)
split<-initial_split(df, .7, strata=Win) #CREATE THE SPLIT
training<-training(split) #TRAINING PARTITION
test<-testing(split) #test PARTITION

#VERIFY STRATIFIED SAMPLING YIELDS EQUALLY SKEWED PARTITIONS
mean(training$Win==1)
mean(test$Win==1)

SVM_Model<- svm(Win ~ OT + Home + PA + RushYd + PassYd + TOA + 
                  OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
                  `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per,
                data = training, 
                type = "C-classification", 
                kernel = kern_type,
                cost=10,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 2,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
(E_IN_PRETUNE<-1-mean(predict(SVM_Model, training)==training$Win))  # 1 - accuracy = error
(E_OUT_PRETUNE<-1-mean(predict(SVM_Model, test)==test$Win))

#TUNING THE SVM BY CROSS-VALIDATION
kern_type <- "radial"
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(3721)
TUNE <- tune.svm(x = training[,-4], #Everything besides column 4
                 y = training[,4], #only column 4
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(training)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE
sum(is.na(training))  # Check if there are any missing values
#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune<- svm(Win ~ OT + Home + PA + RushYd + PassYd + TOA + 
                   OppRush + OppPass + OppTO + comingoffbye + OppPittsburghSteelers + TeamKansasCityChiefs +
                   `3DAtt` + TeamPittsburghSteelers + OppKansasCityChiefs + Conv_per, 
                 data = training, 
                 type = "C-classification", 
                 kernel = kern_type,
                 degree = TUNE$best.parameters$degree,
                 gamma = TUNE$best.parameters$gamma,
                 coef0 = TUNE$best.parameters$coef0,
                 cost = TUNE$best.parameters$cost,
                 scale = FALSE)

print(SVM_Retune) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY) ON RETUNED MODEL
(E_IN_RETUNE<-1-mean(predict(SVM_Retune, training)==training$Win))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune, test)==test$Win))

#SUMMARIZE RESULTS IN A TABLE:
TUNE_TABLE <- matrix(c(E_IN_PRETUNE, 
                       E_IN_RETUNE,
                       E_OUT_PRETUNE,
                       E_OUT_RETUNE),
                     ncol=2, 
                     byrow=TRUE)

colnames(TUNE_TABLE) <- c('UNTUNED', 'TUNED')
rownames(TUNE_TABLE) <- c('E_IN', 'E_OUT')
TUNE_TABLE #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS

#starting classification Tree

#LOADING THE LIBRARIES
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)

df$Win <- as.factor(df$Win)
##PARTITIONING THE DATA##
set.seed(3721)
split<-initial_split(df, prop=.7, strata=Win)
train<-training(split)
test<-testing(split)

#SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- Win ~ . #perfect for NFL Big data model, build regression with everything
class_tree_fit <- fit(class_spec, formula = class_fmla, data = train)

# PRINT THE FITTED MODEL
print(class_tree_fit)
#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree_fit$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)
#plotting
plotcp(class_tree_fit$fit)

# Visualize the decision tree
rpart.plot(class_tree_fit$fit, 
           type = 4,               # Fully labeled tree
           extra = 104,            # Shows classification probability and labels
           box.palette = "RdBu",   # Color scheme for boxes
           shadow.col = "gray",    # Adding shadow to the boxes
           nn = TRUE)              # Displays the number of observations per node
#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class <- predict(class_tree_fit, new_data = test, type="class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

pred_prob <- predict(class_tree_fit, new_data = test, type="prob") %>%
  bind_cols(test) #ADD PROBABILITY PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class$.pred_class, pred_class$Win)
confusionMatrix(confusion, positive='1') #FROM CARET PACKAGE

#GENERATE ROC CURVE AND COMPUTE AUC OVER ALL TRUE / FALSE +'s
autoplot(roc_curve(pred_prob, estimate=.pred_1, truth=Win))
roc_auc(pred_prob, estimate=.pred_1, truth=Win)

library(pROC)
roc_obj <- roc(test$Win, pred_prob$.pred_1)

# Plot the ROC curve
plot(roc_obj, col='blue', main="ROC Curve")

# Compute the AUC (1 minus AUC, as you mentioned the plot is inverted)
auc_value <- 1 - auc(roc_obj)
print(auc_value)

#XGboosted forest 

#LOADING THE LIBRARIES
library(tidymodels)

library(xgboost) #FOR GRADIENT BOOSTING

##PARTITIONING THE DATA##
set.seed(3721)
split<-initial_split(df, prop=.7, strata=Win)
train<-training(split)
test<-testing(split)

#MODEL DESCRIPTION:
fmla <- Win ~.

boosted_forest <- boost_tree(min_n = NULL, #minimum number of observations for split
                             tree_depth = NULL, #max tree depth
                             trees = 100, #number of trees
                             mtry = NULL, #number of predictors selected at each split 
                             sample_size = NULL, #amount of data exposed to fitting
                             learn_rate = NULL, #learning rate for gradient descent
                             loss_reduction = NULL, #min loss reduction for further split
                             stop_iter = NULL)  %>% #maximum iteration for convergence
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(fmla, train)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_xb_in <- predict(boosted_forest, new_data = train, type="class") %>%
  bind_cols(train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_xb_in$.pred_class, pred_class_xb_in$Win)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_xb_out <- predict(boosted_forest, new_data = test, type="class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_xb_out$.pred_class, pred_class_xb_out$Win)
confusionMatrix(confusion) #FROM CARET PACKAGE