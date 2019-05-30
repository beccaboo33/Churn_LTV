#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:17:00 2019

@author: BeccaYiu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc


##Pre-processing
#Reading in and pre-processing data
df = pd.read_csv("Telco_churn.csv")
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors = 'coerce')
df.TotalCharges.fillna(0, inplace = True)
#Binary value
convert_columns = ['Churn', 
                  'Partner',
                  'Dependents',
                  'PhoneService',
                  'PaperlessBilling']
for item in convert_columns:
    df[item].replace(to_replace = "Yes", value = 1, inplace = True)
    df[item].replace(to_replace = "No", value = 0, inplace = True)
user_id = df.customerID #Retrieve ID
df = df.iloc[:, 1:]#Remove ID
df = pd.get_dummies(df)#Creating dummy variables for modeling


labels = pd.Series(df['Churn'])
features = df.drop('Churn', axis = 1)
features_name = list(features.columns)
features = pd.DataFrame(features)

##Modeling
#Splitting data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 37)

#Random forest
rf = RandomForestClassifier()

#Parameter-tuning
n_estimators = [int(x) for x in np.linspace(100, 1000, num = 20)]
max_features = ['auto', 'sqrt']
random_grid_rf = {'n_estimators': n_estimators,
    'max_features': max_features}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 10, cv = 3, random_state = 42)
rf_random.fit(train_features, train_labels)
best_rf_random = rf_random.best_estimator_
best_rf_random.fit(train_features, train_labels)
rf_labels = best_rf_random.predict(test_features)

#Evaluating
fp_rate, tp_rate, thresholds = roc_curve(test_labels, rf_labels)
roc_auc = auc(fp_rate, tp_rate)

#Printing Results
churn_user = pd.DataFrame(rf_labels, index = test_labels.index)
churn_id = user_id.iloc[churn_user.index]
print(churn_id)
print(roc_auc)
