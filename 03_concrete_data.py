#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ """

import pandas as pd

names= ['Cement','BFS','FLA','Water','SP','CA','FA','Age','CCS']
Data = pd.read_excel('ConcreteData.xlsx', names=names)

Data.head(10)
Data.info()
Data.describe()

import seaborn as sns
sns.set(style="ticks")
sns.boxplot(data = Data)

sns.pairplot(data = Data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(Data))
data_scaled = scaler.fit_transform(Data)
data_scaled = pd.DataFrame(data_scaled, columns=names)

data_scaled.describe()
sns.boxplot(data = data_scaled)

from sklearn.model_selection import train_test_split
Predictors = pd.DataFrame(data_scaled.iloc[:,:8])
Response = pd.DataFrame(data_scaled.iloc[:,8])

Pred_train, Pred_test, Resp_train, Resp_test = train_test_split(Predictors,Response, test_size = 0.30, random_state = 1)
Pred_train.shape,Pred_test.shape, Resp_train.shape, Resp_test.shape

# Keras Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(Pred_train, Resp_train, epochs=100, verbose=1)
#model.fit(Pred_train, Resp_train, epochs=1000, verbose=1)

model.summary()

Y_predKM = model.predict(Pred_test)

from sklearn.metrics import r2_score
print('Coefficient of determination of Keras Model:', r2_score(Resp_test, Y_predKM))
#print(r2_score(Resp_test, Y_predKM))

Q1 = data_scaled.quantile(0.25)
Q3 = data_scaled.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

data_scaled_out = data_scaled[~((data_scaled < (Q1 - 1.5 * IQR)) | (data_scaled > (Q3 + 1.5 * IQR))).any(axis=1)]
data_scaled_out.shape

import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
sns.boxplot(data = data_scaled)
plt.subplot(122)
sns.boxplot(data = data_scaled_out)

## Model 2
Predictors2 = pd.DataFrame(data_scaled_out.iloc[:,:8])
Response2 = pd.DataFrame(data_scaled_out.iloc[:,8])

Pred_train2, Pred_test2, Resp_train2, Resp_test2 = train_test_split(Predictors2,Response2, test_size = 0.30, random_state = 1)
Pred_train2.shape, Pred_test2.shape, Resp_train2.shape, Resp_test2.shape

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(Pred_train2, Resp_train2, epochs=100, verbose=1)
#model.fit(Pred_train2, Resp_train2, epochs=1000, verbose=1)

model.summary()

Y_predKM2 = model.predict(Pred_test2)

print('Coefficient of determination of Keras Model')
print(r2_score(Resp_test, Y_predKM))

print('Coefficient of determination of Keras Model without outlier')
print(r2_score(Resp_test2, Y_predKM2))


plt.figure(1)
plt.subplot(121)
plt.scatter(Resp_test,Y_predKM)
plt.plot([0, 1], [0, 1], linewidth=2)
plt.subplot(122)
plt.scatter(Resp_test2, Y_predKM2)
plt.plot([0, 1], [0, 1], linewidth=2)
