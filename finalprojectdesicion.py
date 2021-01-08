# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:52:58 2021

@author: HaticeOzdemir
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

col_names = ['fixed acidity','volatile acidity','citric acid',	'residual sugar','chlorides', 'free sulfur dioxide',	'total sulfur dioxide'	,'density', 'pH',	'sulphates',	'alcohol',	'quality']

veriler=pd.read_csv('C:/Users/HaticeOzdemir/Desktop/opencvdersleri/winequality.csv')

feature_cols = ['fixed acidity','volatile acidity','citric acid',	'residual sugar','chlorides', 'free sulfur dioxide',	'total sulfur dioxide'	,'density', 'pH',	'sulphates',	'alcohol']
print(veriler.head())
X = veriler[feature_cols] # Features
y = veriler.quality # Target variable


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier=classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



