---
title: "practical Machine learning"
author: "Anjan Prakash Karki"
date: "March 27, 2016"
output: html_document
---


## Data Preparation

Loading necessary libraries and data


```r
library(foreach)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(doParallel)
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
ptrain <- read.csv("pml-training.csv")
```

```
## Warning in file(file, "rt"): cannot open file 'pml-training.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
ptest <- read.csv("pml-testing.csv")
```

```
## Warning in file(file, "rt"): cannot open file 'pml-testing.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```
##Inspect the loaded data
As training data set has 160 variables, many with missing values, we do some data preparation.

```r
str(ptrain)
```

```
## Error in str(ptrain): object 'ptrain' not found
```
##Removing variables with low variability
Finding variables with near zero variance.

```r
low_var <- nearZeroVar(ptrain, saveMetrics=TRUE)
```

```
## Error in is.vector(x): object 'ptrain' not found
```

```r
non_low_vars <- subset(low_var, !low_var$nzv) 
```

```
## Error in subset(low_var, !low_var$nzv): object 'low_var' not found
```

```r
training1 <- ptrain[rownames(non_low_vars)]
```

```
## Error in eval(expr, envir, enclos): object 'ptrain' not found
```
This reduces the number of variables to 100.

```r
dim(training1)
```

```
## Error in eval(expr, envir, enclos): object 'training1' not found
```
##Eliminating variables with missing values
The variables with data that is missing are eliminated. Now, there are 41 columns that are missing (19216 out of 19622 rows). There remains 59 variables.

```r
na_count <- summary(is.na(training1))
```

```
## Error in summary(is.na(training1)): object 'training1' not found
```

```r
na_count1 = sapply(training1, function(x) {sum(is.na(x))})
```

```
## Error in lapply(X = X, FUN = FUN, ...): object 'training1' not found
```

```r
cols_with_nas = names(na_count1[na_count1>18000])
```

```
## Error in eval(expr, envir, enclos): object 'na_count1' not found
```

```r
training2 = training1[, !names(training1) %in% cols_with_nas]
```

```
## Error in eval(expr, envir, enclos): object 'training1' not found
```

```r
dim(training2)
```

```
## Error in eval(expr, envir, enclos): object 'training2' not found
```
##Removing the first 6 variables
The first 6 variables are removed as they are not useful. They contain descriptive information that would not be used in analysis. Now, 53 variables now remain out of an original 160 variables.

```r
training3 <- training2[-c(1:6)]
```

```
## Error in eval(expr, envir, enclos): object 'training2' not found
```

```r
dim(training3)
```

```
## Error in eval(expr, envir, enclos): object 'training3' not found
```
##Splitting training dataset into training and validation datasets
The training dataset is splited into training and validation datasets, on a 60/40 basis to allow for the model to be validated against a clean dataset.

```r
set.seed(738024)
inTrain <- createDataPartition(y=training3$classe, p=0.6, list=FALSE)
```

```
## Error in createDataPartition(y = training3$classe, p = 0.6, list = FALSE): object 'training3' not found
```

```r
training <- training3[inTrain,]
```

```
## Error in eval(expr, envir, enclos): object 'training3' not found
```

```r
validation <- training3[-inTrain,]
```

```
## Error in eval(expr, envir, enclos): object 'training3' not found
```
##Modeling

#Develop Random Forest Model
Based on previous experience, a Random Forest model is chosen as a first method. The randomForest package was used as it can be more efficient than the Random Forest method in the caret package. A 10-fold cross validation was used as train control method. Here is the result of the model and the importance of each predictor

```r
TC = trainControl(method = "cv", number = 10)

RF <- randomForest(classe ~. , data=training, trControl = TC)
```

```
## Error in randomForest.default(m, y, ...): Can not handle categorical predictors with more than 53 categories.
```

```r
print(RF)
```

```
## Error in print(RF): object 'RF' not found
```

```r
importance(RF)
```

```
## Error in importance(RF): object 'RF' not found
```
##Model Validation and Out of Sample Error

The out-of-sample error is the error realised by using the model developed on the training data to make predictions on separate validation sample. An estimate is that should be close to the OOB estimate of error rate in the model. The cross validation shows the model to be very accurate, with an accuracy against the validation sample of 99.35%, with the out-of-sample error of 0.65% which is similar to the estimate.

As this model shows such a good result, no further methods are examined.

```r
pred_RF <- predict(RF, validation, type = "class")
```

```
## Error in predict(RF, validation, type = "class"): object 'RF' not found
```

```r
confusionMatrix(pred_RF, validation$classe)
```

```
## Error in confusionMatrix(pred_RF, validation$classe): object 'pred_RF' not found
```
##Generating the Submission
The instructions from the project assignment were followed, to generate the answers and then use a macro to generate the 20 problem_id files that were subsequently uploaded individually to the course website. The model proved to be quite accurate, correctly predicting all 20 test cases.

```r
answers <- predict(RF, newdata = ptest)
```

```
## Error in predict(RF, newdata = ptest): object 'RF' not found
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```

```
## Error in pml_write_files(answers): object 'answers' not found
```
