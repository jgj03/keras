#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ """

import pandas as pd

#Import data
HDNames= ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','hal','HeartDisease']
Data = pd.read_excel('ClevelandData.xlsx', names=HDNames)
print(Data.head(10))
Data.info()
Data.describe()


#Removing missing values
import numpy as np
Data.isin(['nan']).sum()

datanew = Data.replace('?', np.nan)
print(datanew.info())
print(datanew.describe())

print(datanew.isnull().sum())

data = datanew.dropna()
data.info()
data.isnull().sum()
data.columns

#Divide DataFrame
InputNames = HDNames
InputNames.pop()
X = pd.DataFrame(data.iloc[:, 0:13],columns=InputNames)
y = pd.DataFrame(data.iloc[:, 13],columns=['HeartDisease'])

#Data scaling
from sklearn.preprocessing import StandardScaler
#Standardize features by removing the mean and scaling to unit variance. 
scaler = StandardScaler()
print(scaler.fit(X))
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=InputNames)

summary = X_scaled.describe()
summary = summary.transpose()
print(summary)

# Data visualitation
#DataScaled = pd.concat([InputScaled, Target], axis=1)
import matplotlib.pyplot as plt
boxplot = X_scaled.boxplot(column=InputNames,showmeans=True, figsize=(7,7))
plt.show()

pd.plotting.scatter_matrix(X_scaled, figsize=(7, 7))
plt.show()

CorData = X_scaled.corr(method='pearson')

with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)

plt.matshow(CorData)
plt.xticks(range(len(CorData.columns)), CorData.columns)
plt.yticks(range(len(CorData.columns)), CorData.columns)
plt.colorbar()
plt.show()

#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.30, random_state = 5)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Keras Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=13, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)
#model.fit(X_train, y_train, epochs=1000, verbose=1)

model.summary()

score = model.evaluate(X_test, y_test, verbose=0)
print('Keras Model Accuracy = ',score[1])

y_Classification = model.predict(X_test)
y_Classification = (y_Classification > 0.5)

from sklearn.metrics import confusion_matrix  #[tn fp 
cm=confusion_matrix(y_test, y_Classification) #fn,tp]
print(cm)
tn,fp,fn,tp = cm.ravel() #tn,fp,fn,tp
print('True Negative:{}, False Positive:{}, False Negative:{}, True Positive:{}'.format(tn,fp,fn,tp))
plt.imshow(cm, cmap='binary')

import seaborn as sns
sns.heatmap(cm)
sns.set(font_scale=1.4)#for label size
sns.heatmap(cm, annot=True,annot_kws={"size": 16})

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
