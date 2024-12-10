df <-NFl_big_data_cleaned
View(df)
df$Conv_per <-(df$`3DConv`/df$`3DAtt`)    #creating 3rd down conversion rate
View(df)
df$Conv_per <- (df$Conv_per)*100 #coverting to a percentage 
View(df)
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



#Classification
# Load necessary libraries
library(caret)
library(dplyr)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071) 

# For SVM

# Load the data
nfl_data <- read.csv("NFl_big_data_cleaned.csv")

# Set seed for reproducibility
set.seed(123)

#Partitioning the Data
cat("Partitioning Data...\n")
train_index <- createDataPartition(nfl_data$Win, p = 0.7, list = FALSE)
train_data <- nfl_data[train_index, ]
remaining_data <- nfl_data[-train_index, ]
validation_index <- createDataPartition(remaining_data$Win, p = 0.5, list = FALSE)
validation_data <- remaining_data[validation_index, ]
test_data <- remaining_data[-validation_index, ]

# Save partitions for reproducibility
write.csv(train_data, "train_data.csv", row.names = FALSE)
write.csv(validation_data, "validation_data.csv", row.names = FALSE)
write.csv(test_data, "test_data.csv", row.names = FALSE)

cat("Training set size:", nrow(train_data), "\n")
cat("Validation set size:", nrow(validation_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

#Logistic Regression Model
cat("\nFitting Logistic Regression Model...\n")
logistic_model <- glm(Win ~ ., data = train_data, family = "binomial")
summary(logistic_model)

validation_preds <- predict(logistic_model, newdata = validation_data, type = "response")
validation_preds_class <- ifelse(validation_preds > 0.5, 1, 0)
validation_accuracy <- mean(validation_preds_class == validation_data$Win)
cat("Logistic Regression Validation Accuracy:", validation_accuracy, "\n")

# ROC Curve and AUC
roc_curve <- roc(validation_data$Win, validation_preds)
plot(roc_curve, main = "ROC Curve for Logistic Regression")
cat("Logistic Regression AUC:", auc(roc_curve), "\n")

# Probit Model
cat("\nFitting Probit Model...\n")
probit_model <- glm(Win ~ ., data = train_data, family = binomial(link = "probit"))
summary(probit_model)

probit_preds <- predict(probit_model, newdata = validation_data, type = "response")
probit_preds_class <- ifelse(probit_preds > 0.5, 1, 0)
probit_accuracy <- mean(probit_preds_class == validation_data$Win)
cat("Probit Model Validation Accuracy:", probit_accuracy, "\n")

# Compare Probit and Logit AUC
probit_roc <- roc(validation_data$Win, probit_preds)
cat("Probit Model AUC:", auc(probit_roc), "\n")

# Decision Tree
cat("\nFitting Decision Tree Model...\n")
tree_model <- rpart(Win ~ ., data = train_data, method = "class")
rpart.plot(tree_model)

tree_preds <- predict(tree_model, newdata = validation_data, type = "class")
tree_accuracy <- mean(tree_preds == validation_data$Win)
cat("Decision Tree Validation Accuracy:", tree_accuracy, "\n")

# Random Forest
cat("\nFitting Random Forest Model...\n")
rf_model <- randomForest(Win ~ ., data = train_data, ntree = 100)

rf_preds <- predict(rf_model, newdata = validation_data)
rf_accuracy <- mean(rf_preds == validation_data$Win)
cat("Random Forest Validation Accuracy:", rf_accuracy, "\n")

# Support Vector Machine
cat("\nFitting Support Vector Machine...\n")
svm_model <- svm(Win ~ ., data = train_data, probability = TRUE)
svm_preds <- predict(svm_model, newdata = validation_data, probability = TRUE)
svm_accuracy <- mean(svm_preds == validation_data$Win)
cat("SVM Validation Accuracy:", svm_accuracy, "\n")

# Compare Results
cat("\nSummary of Validation Accuracies:\n")
results <- data.frame(
  Model = c("Logistic Regression", "Probit Model", "Decision Tree", "Random Forest", "SVM"),
  Validation_Accuracy = c(validation_accuracy, probit_accuracy, tree_accuracy, rf_accuracy, svm_accuracy)
)
print(results)

# Plot ROC for all models
plot(roc_curve, col = "blue", main = "ROC Curves for Models")
plot(probit_roc, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Probit"), col = c("blue", "red"), lty = 1)
