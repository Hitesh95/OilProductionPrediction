
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import ExcelWriter
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot



dataset = pd.read_csv("C:/Users/h.rajesh.hinduja/Desktop/new_file.csv")
dataset
x = np.array(dataset)
x
X = x[:,0:17]
Y = x[:,17]
Y
seed = 8
test_size = 0.15
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y ,test_size=test_size, random_state=seed)
y_test

xgb = xgboost.XGBRegressor(n_estimators=135, learning_rate=0.04, gamma=0, subsample=0.9,
                           colsample_bytree=0.5, max_depth=10,max_delta_step=0,min_child_weight=1,colsample_bylevel=0.8,reg_lambda=1, reg_alpha=0, booster='gbtree', base_score=0.5)
traindf, testdf = train_test_split(X_train, test_size = test_size)
xgb.fit(X_train,y_train)



predictions = xgb.predict(X_test)
predictions
rms = sqrt(mean_squared_error(y_test, predictions))
print(rms)



df = pd.DataFrame(predictions, y_test)
df

df.to_csv('C:/Users/h.rajesh.hinduja/Desktop/PythonExport4.csv', sep=',')

predictions
print(explained_variance_score(y_test,predictions))
predictions

print(xgb.feature_importances_)

pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()


plot_importance(xgb)
pyplot.show()


