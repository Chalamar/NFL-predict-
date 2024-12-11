# Load necessary libraries
library(caret)
library(dplyr)
library(ggplot2)
library(glmnet)
library(randomForest)
library(mgcv)

# Load the data
df <- read.csv("NFl_big_data_cleaned.csv") 

# Set seed
set.seed(126)

# Define target and features
target <- "PF"  # Replace with your continuous target
features <- setdiff(names(df), target)

# Partition data
train <- createDataPartition(df[[target]], p = 0.7, list = FALSE)
train_data <- df[train, ]
temp_data <- df[-train, ]
validation_index <- createDataPartition(temp_data[[target]], p = 0.5, list = FALSE)
validation_data <- temp_data[validation_index, ]
test_data <- temp_data[-validation_index, ]

# Bivariate Regression
cat("\n--- Bivariate Regression ---\n")
b_model <- lm(as.formula(paste(target, "~", features[1])), data = train_data)
b_preds <- predict(b_model, validation_data)
b_rmse <- sqrt(mean((validation_data[[target]] - b_preds)^2))
b_r2 <- cor(validation_data[[target]], b_preds)^2
cat("Bivariate Regression - RMSE:", bivariate_rmse, "R²:", bivariate_r2, "\n")

# Multivariate Linear Regression
cat("\n--- Multivariate Linear Regression ---\n")
l_model <- lm(as.formula(paste(target, "~ .")), data = train_data)
l_preds <- predict(l_model, validation_data)
l_rmse <- sqrt(mean((validation_data[[target]] - l_preds)^2))
l_r2 <- cor(validation_data[[target]], l_preds)^2
cat("Linear Regression - RMSE:", l_rmse, "R²:", l_r2, "\n")

# LASSO Regression
cat("\n--- LASSO Regression ---\n")
x_train <- model.matrix(as.formula(paste(target, "~ .")), data = train_data)[, -1]
y_train <- train_data[[target]]
x_validation <- model.matrix(as.formula(paste(target, "~ .")), data = validation_data)[, -1]
y_validation <- validation_data[[target]]

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)  # LASSO
lasso_preds <- predict(lasso_model, s = "lambda.min", newx = x_validation)
lasso_rmse <- sqrt(mean((y_validation - lasso_preds)^2))
lasso_r2 <- cor(y_validation, lasso_preds)^2
cat("LASSO Regression - RMSE:", lasso_rmse, "R²:", lasso_r2, "\n")

# GAM
cat("\n--- Generalized Additive Model (GAM) ---\n")
gam_model <- gam(as.formula(paste(target, "~ s(", features[1], ")")), data = train_data)
gam_preds <- predict(gam_model, validation_data)
gam_rmse <- sqrt(mean((validation_data[[target]] - gam_preds)^2))
gam_r2 <- cor(validation_data[[target]], gam_preds)^2
cat("GAM - RMSE:", gam_rmse, "R²:", gam_r2, "\n")

# Random Forest
cat("\n--- Random Forest Regression ---\n")
rf_model <- randomForest(as.formula(paste(target, "~ .")), data = train_data, ntree = 100)
rf_preds <- predict(rf_model, validation_data)
rf_rmse <- sqrt(mean((validation_data[[target]] - rf_preds)^2))
rf_r2 <- cor(validation_data[[target]], rf_preds)^2
cat("Random Forest - RMSE:", rf_rmse, "R²:", rf_r2, "\n")

# Summary of Results
cat("\n--- Summary of Results ---\n")
results <- data.frame(
  Model = c("Bivariate Regression", "Linear Regression", "LASSO", "GAM", "Random Forest"),
  RMSE = c(bivariate_rmse, linear_rmse, lasso_rmse, gam_rmse, rf_rmse),
  R2 = c(bivariate_r2, linear_r2, lasso_r2, gam_r2, rf_r2)
)
print(results)

# Evaluate Best Model on Test Set (Random Forest in this case)
cat("\n--- Final Model Evaluation on Test Set ---\n")
final_preds <- predict(rf_model, test_data)
test_rmse <- sqrt(mean((test_data[[target]] - final_preds)^2))
test_r2 <- cor(test_data[[target]], final_preds)^2
cat("Final Model (Random Forest) - Test RMSE:", test_rmse, "R²:", test_r2, "\n")

# Visualizations (GAM example)
plot(gam_model, residuals = TRUE, main = "GAM: Residuals vs Predictor")
