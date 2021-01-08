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

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)


k_range=range(1,30)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test,y_test))
    
plt.figure()
plt.xlabel(k)
plt.ylabel('dogruluk')
plt.scatter(k_range,scores)
plt.xticks([0,5,10,15,20]);


from sklearn.metrics import classification_report, confusion_matrix
cmatrix=confusion_matrix(y_test,y_pred)
print(cmatrix)
print(classification_report(y_test,y_pred))
toplam=sum(sum(cmatrix))
accuracy=(cmatrix[0,0]+cmatrix[1,1])/toplam
print('accuracy',accuracy)
sensitivity=cmatrix[0,0]/(cmatrix[0,0]+cmatrix[0,1])
print('sensitivty:',sensitivity)
specificty=cmatrix[1,1]/(cmatrix[1,0]+cmatrix[1,1])
print('specificty:',specificty)




