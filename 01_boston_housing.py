#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ """

import pandas as pd
BHNames= ['crim','zn','indus','chas','nox','rm', 'age','dis','rad','tax','ptratio','black','lstat','medv']

url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
data = pd.read_csv(url, delim_whitespace=True, names=BHNames)

print(data.head(5))
print(data.info())

data.describe()
summary = data.describe()
summary = summary.transpose()
print(summary)


from sklearn.preprocessing import MinMaxScaler
#Transforms features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.
scaler = MinMaxScaler()
print(scaler.fit(data))
DataScaled = scaler.fit_transform(data)
DataScaled = pd.DataFrame(DataScaled, columns=BHNames)

summary = DataScaled.describe()
summary = summary.transpose()
print(summary)

import matplotlib.pyplot as plt
boxplot = DataScaled.boxplot(column=BHNames)
plt.show()

CorData = DataScaled.corr(method='pearson')
with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)

plt.matshow(CorData)
plt.xticks(range(len(CorData.columns)), CorData.columns)
plt.yticks(range(len(CorData.columns)), CorData.columns)
plt.colorbar()
plt.show()


data.columns #Medv is the output
X = DataScaled.drop('medv', axis = 1)
print('X shape = ',X.shape)
y = DataScaled['medv']
print('Y shape = ',y.shape)
data.shape, X.shape, y.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 5)
print('X train shape = ',X_train.shape)
print('X test shape = ', X_test.shape)
print('Y train shape = ', y_train.shape)
print('Y test shape = ', y_test.shape)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#Keras Model
from keras.models import Sequential
from keras.layers import Dense
#from keras import metrics

model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)
#model.fit(X_train, y_train, epochs=1000, verbose=1)

model.summary()

y_predKM = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0)

print('Keras Model')
print(score[0])


#Linear Regression
from sklearn.linear_model import LinearRegression

LModel = LinearRegression()
LModel.fit(X_train, y_train)

y_predLM = LModel.predict(X_test)

plt.figure(1)
plt.subplot(121)
plt.scatter(y_test, y_predKM)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Keras Neural Network Model")

plt.subplot(122)
plt.scatter(y_test, y_predLM)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("SKLearn Linear Regression Model")
plt.show()

from sklearn.metrics import mean_squared_error
mse_lr = mean_squared_error(y_test, y_predLM)
print('Linear Regression Model: ',mse_lr)

mse_krs = mean_squared_error(y_test, y_predKM)
print('Keras NN Model: ',mse_krs)

