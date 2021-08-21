import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks.callbacks import ModelCheckpoint

tesla = pd.read_csv('TSLA.csv')

tesla = tesla[['Date', 'Close']]
new_tesla = tesla.loc[884:1639]
new_tesla = new_tesla.drop('Date', axis=1)
new_tesla = new_tesla.reset_index(Drop=True)

T = T.astype('float32')
T = np.reshape(T, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
T = scaler.fit_transform(T)

train_size = int(len(T) * 0.80)
test_size = int(len(T) - train_size)
train, test = T[0:train_size, :], T[train_size:len(T), :]

# create 20 day vector
def create_features(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size - 1:
        window = data[i: (i + window_size), 0]
        X.append(window)
        Y.append(data[i + window_size, 0])
    return(np.array(X), np.array(Y))

window_size = 20
X_train, Y_train = create_features(train, window_size)
X_test, Y_test = create_features(test, window_size)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def isLeak(T_shape, train_shape, test_shape):
    return not(T_shape[0] == (train_shape[0] + test_shape[0]))

print(isLeak(T_shape, train_shape, test_shape))


#adding the lstm layer to the model
tf.random.set_seed(11)
np.random.seed(11)
model = Sequential
model.add(LSTM(units = 50, activation = 'relu', input_shape = (X_train.shape[1], window_size)))
model.add(Dropout(0.2))
model.add(Dense(1))

# 8:26 
# https://www.youtube.com/watch?v=SauRW1Vok44&ab_channel=Kite
# https://github.com/kiteco/python-youtube-code/blob/master/predicting-tesla-stock-price/predict-tesla-stock-price-LSTM.py