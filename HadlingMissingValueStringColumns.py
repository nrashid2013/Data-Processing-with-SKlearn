# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:18:44 2019

@author: nrashid
"""

import numpy
import pandas as pd

df = pd.read_csv('TestData.csv')
df
X_axis = df.iloc[:,:-1].values
y_axis = df.iloc[:,3].values

#handleing missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X_axis[:, 1:3])
X_axis[:, 1:3]=imputer.transform(X_axis[:, 1:3])


#handleing categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X_axis[:,0]=labelencoder.fit_transform(X_axis[:,0])


onehotencoder = OneHotEncoder(categorical_features=[0])
X_axis=onehotencoder.fit_transform(X_axis).toarray()

