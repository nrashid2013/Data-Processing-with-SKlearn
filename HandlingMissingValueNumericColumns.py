# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:18:44 2019

@author: nrashid
"""

import numpy
import pandas as pd

df = pd.read_csv('TestData.csv')
df = pd.DataFrame(df)
X = df.iloc[:,:-1].values
y = df.iloc[:,3].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
X = pd.DataFrame(X)
y = pd.DataFrame(y)
