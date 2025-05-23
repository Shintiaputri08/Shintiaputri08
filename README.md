title: "Practical Machine Learning: Prediction Assignment Writeup"
author: "Shintia Lestari Putri"
date: "`r Tanggal sistem ()`"
output: html_document

# Introduction

This project aims to predict how participants performed barbell lifts using accelerometer data collected from six individuals wearing sensors on the belt, forearm, arm, and dumbbell. The target variable is **Klasifikasi**, which categorizes five different ways the barbell lifts were executed (correctly or incorrectly).

The data source is the Weight Lifting Exercise Dataset from the [PUC-Rio Groupware site](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

# Data Loading and Preprocessing

The training and test datasets were loaded from Coursera’s provided URLs. Initial data cleaning involved removing variables with a large proportion of missing values (over 95%), eliminating identifiers and near-zero variance predictors to reduce noise and improve model performance.

```{r load-data, echo=TRUE, message=FALSE, warning=FALSE}
library(caret)
library(dplyr)

# Load training data
train_raw <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

# Check missing data proportions
missing_prop <- sapply(train_raw, function(x) mean(is.na(x)))

# Remove columns with >95% missing
train_clean <- train_raw[, missing_prop <= 0.95]

# Remove non-predictor columns (IDs, timestamps, etc.)
train_clean <- train_clean[, -c(1:7)]

# Remove near zero variance predictors
nzv <- nearZeroVar(train_clean)
if(length(nzv) > 0){
  train_clean <- train_clean[, -nzv]
}

# Remove any columns with remaining NA values
train_clean <- train_clean[, colSums(is.na(train_clean)) == 0]

# Inspect final dimensions
dim(train_clean)
```

# Data Partitioning

To evaluate model performance, the cleaned data was split into training (70%) and validation (30%) subsets using stratified sampling on the outcome variable.

```{r partition-data, echo=TRUE}
set.seed(123)
inTrain <- createDataPartition(y = train_clean$classe, p = 0.7, list = FALSE)
training <- train_clean[inTrain, ]
validation <- train_clean[-inTrain, ]
```

# Model Building: Random Forest

Random Forest was chosen for its high accuracy and ability to handle many predictor variables without overfitting easily. Five-fold cross-validation was used to estimate out-of-sample error.

```{r train-model, echo=TRUE, message=FALSE}
fitControl <- trainControl(method = "cv", number = 5)
set.seed(123)
rf_model <- train(classe ~ ., data = training, method = "rf", trControl = fitControl)
rf_model
```

# Model Performance on Validation Set

The model’s accuracy and confusion matrix on the validation set were examined to assess performance.

```{r validate-model, echo=TRUE}
validation_pred <- predict(rf_model, validation)
conf_matrix <- confusionMatrix(validation_pred, validation$classe)
conf_matrix
```

The model achieved an accuracy of approximately `r round(conf_matrix$overall['Accuracy'] * 100, 2)`% on the validation data, indicating strong predictive performance.

# Final Predictions on Test Data

The test set was loaded and cleaned by selecting only the columns used in training, excluding the target variable. The final model was used to predict the exercise class for each of the 20 test cases.

```{r predict-test, echo=TRUE}
test_raw <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))

# Select only variables used in training (except classe)
test_clean <- test_raw[, colnames(test_raw) %in% colnames(training)]
test_clean <- test_clean[, -which(names(test_clean) == "classe")]

# Generate predictions
test_predictions <- predict(rf_model, test_clean)
test_predictions
```

# Save Predictions to Files for Submission

The following function saves each of the 20 predictions as individual text files named `problem_id_1.txt`, `problem_id_2.txt`, ..., inside a folder called `predictions`.

```{r save-predictions, echo=TRUE}
save_predictions_files <- function(predictions) {
  if(!dir.exists("predictions")) {
    dir.create("predictions")
  }
  
  for(i in seq_along(predictions)) {
    filename <- paste0("predictions/problem_id_", i, ".txt")
    write.table(predictions[i], file = filename, quote = FALSE,
                row.names = FALSE, col.names = FALSE)
  }
  message("20 prediction files saved in ./predictions/")
}

save_predictions_files(test_predictions)
```

# Cross-Validation and Expected Out-of-Sample Error

Using 5-fold cross-validation during training provides a robust estimate of the model's expected performance on new, unseen data, helping to prevent overfitting. The validation set accuracy corroborates this estimate.

# Conclusion

The Random Forest model, combined with careful data cleaning and cross-validation, produced highly accurate predictions for the exercise classification task. The predictions on the test set have been saved in separate files ready for submission.

---
*Report submitted by Shintia Lestari Putri on `r Sys.Date()`*
