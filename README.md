# OilProductionPrediction
1 Executive Summary  and 2 Introduction
        - This project predicts the oil production till date. The data is from 1992 till present and initially we have 16 parameters which are finally reduced to 4 parameters and we can predict the oil production. The below given is the index of the steps followed to build this predictive model.

3 Loading and Exploring Data 
3.1 Loading libraries required and reading the data into R
3.2 Data size and structure

4 Exploring some of the most important variables 
4.1 The response variable;
4.2 The most important numeric predictors 
4.2.1 Correlations with response variable that is oil production rate

5 Missing data, label encoding, and factorizing variables 
5.1 Completeness of the data
5.2 Imputing missing data 
5.3 Label encoding/factorizing the remaining character variables 
5.4 Changing some numeric variables into factors

6 Visualization of important variables 
6.1 Correlations again
6.2 Finding variable importance with a quick Random Forest

7 Preparing data for modeling 
7.1 Dropping highly correlated variables
7.2 Removing outliers
7.3 PreProcessing predictor variables 
7.3.1 Skewness and normalizing of the numeric predictors
7.3.2 One hot encoding the categorical variables
7.3.3 Removing levels with few or no observations in train or test
7.4 Dealing with skewness of response variable
7.5 Composing train and test sets

8.3 Averaging predictions
