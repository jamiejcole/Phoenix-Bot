#https://www.youtube.com/watch?v=QIUxPv5PJOY

#approx video time: 25:52


# import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-01-01')
print(df)

df.shape

plt.figure(figsize=(16,8))
plt.title('Close price history of AAPL')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# new df with only close data
data = df.filter(['Close'])

#convert to numpy array
dataset = data.values

#get number of rows to train model on
training_data_len = math.ceil(len(dataset) * .8)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# create the (scaled?) training data set
train_data = scaled_data[0:training_data_len, :]

#split data into x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

  if i <= 61:
    print(x_train)
    print(y_train)
    print()

#convert training to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

#create test dataset
#create array w/ scaled vals from 1543, 2003
test_data = scaled_data[training_data_len - 60: , :]
#create dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)

# get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

# plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualise the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(train['Close'[-50:]])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Getting the predicted price for tomorrow :)

#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-07-14')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the last 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set  to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price_scaled = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price_scaled)
print("Todays predicted price: ", pred_price)

#Getting the predicted price for the day after tomorrow (including tomorrows prediction)
#Appends the predicted price for the next day to the last 60 days
last_60_days_scaled = np.append(last_60_days_scaled, pred_price_scaled, axis = 0)
last_60_days_scaled = np.delete(last_60_days_scaled, 1, axis=0)
X_test = []
#Append the last 61 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set  to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price_scaled = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price_scaled)
print("Tomorrows predicted price: ", pred_price)

apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2021-07-15', end='2021-07-16')
print(apple_quote2['Close'])