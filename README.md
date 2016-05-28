# Santanderbank
The problem is to predict whether a customer is satisfied ot dissatisfied. 

Feature selection: </br>
1.336 features are given in the dataset </br>
2.29 features are removed because they are duplicate or have zero variance </br>
3. Extratree classifier is used to select the features of importance. 71 columns are selected based on this

Building Model: </br>
Logistic regression model , classification Tree, random forest model and gradient boosting models are built on the data. For each model insample and out of sample AUC is measured. Boosting model is selected because of best predictive accuracy on out of sample.AUC on out-if-sample is 0.83
