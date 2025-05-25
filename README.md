Title: "Practical Machine Learning: Prediction Assignment Writeup: From dataset barbell lifts"
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
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Klasifikasi Aktivitas Angkat Barbel dengan Random Forest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notebook ini melanjutkan eksplorasi data dengan melakukan klasifikasi menggunakan Random Forest.\n",
    "\n",
    "Pastikan file dataset (`pml-training.csv`) sudah ada di folder `data/`."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Import library\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load dan Bersihkan Data"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Load data\n",
    "df = pd.read_csv('../data/pml-training.csv')\n",
    "print(f\"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}\")"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Hapus kolom dengan missing value > 90%\n",
    "threshold = 0.9 * len(df)\n",
    "df = df.dropna(axis=1, thresh=threshold)\n",
    "\n",
    "# Hapus kolom metadata\n",
    "cols_to_drop = ['X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', \n",
    "                'cvtd_timestamp', 'new_window', 'num_window']\n",
    "df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')\n",
    "\n",
    "print(f\"Setelah pembersihan: {df.shape[0]} baris, {df.shape[1]} kolom\")"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pisahkan Fitur dan Target"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Target (kelas) adalah kolom 'classe'\n",
    "X = df.drop(columns=['classe'])\n",
    "y = df['classe']\n",
    "\n",
    "print(X.shape, y.shape)"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Split Data: Train & Test"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(X_train.shape, X_test.shape)"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "rf.fit(X_train, y_train)\n",
    "print('Training selesai!')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluasi Model"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "y_pred = rf.predict(X_test)\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "print(f'Akurasi pada data test: {acc:.4f}')\n",
    "print('\\nClassification Report:')\n",
    "print(classification_report(y_test, y_pred))"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Confusion Matrix\n",
    "plt.figure(figsize=(8,6))\n",
    "cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)\n",
    "sns.heatmap(cm, annot=True, fmt='d', xticklabels=rf.classes_, yticklabels=rf.classes_, cmap='Blues')\n",
    "plt.xlabel('Prediksi')\n",
    "plt.ylabel('Aktual')\n",
    "plt.title('Confusion Matrix Random Forest')\n",
    "plt.show()"
   ],
   "execution_count": null,
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

---
*Report submitted by Shintia Lestari Putri on `r Sys.Date()`*
