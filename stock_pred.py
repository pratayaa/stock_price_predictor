# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:31:12 2019

@author: PRATAYA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]

# reshaping         (1257-, 1- timestep, 1- no. of features)
X_train = np.reshape(X_train, (1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# initialize the RNN
regressor = Sequential()

# adding the input lawyer and LSTM (in input shape node means any time step, 1 - means no of features)
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
# output lawyer
regressor.add(Dense(units = 1))
 # compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit( X_train, y_train, batch_size = 32, epochs = 200)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'real google srock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predicted google stock price')
plt.title('google stock prediction')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()



































 