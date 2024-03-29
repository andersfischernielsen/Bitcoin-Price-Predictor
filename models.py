from numpy import sqrt

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import backend as K

import matplotlib.pyplot as plt

# Read
data = read_csv("D1.csv", header=0, index_col=0, skiprows=0)
values = data.values.astype('float64')
# Remove data that would not be known at prediction time ("high", "low", "close")
values = values[:, 0:4:]

# Split data into test, train and validation with validation as most recent data
percent = int(values.shape[0]*0.01)
train = values[:percent*66, :]
val = values[percent*23:percent*90, :]
test = values[percent*90:percent*100, :]

# Split and scale data into train, val test sets
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)
test = scaler.fit_transform(test)

train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# Reshape input for LSTM into [samples, timesteps, features]
train_X_lstm = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X_lstm = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X_lstm = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# SVR
svr = SVR()
svr.fit(train_X, train_y)

# LR
lr = LinearRegression()
lr.fit(train_X, train_y)

# LSTM
lstm_model = Sequential()
units = int(train_X_lstm.shape[0]/10)
lstm_model.add(LSTM(units, input_shape=(
    train_X_lstm.shape[1], train_X_lstm.shape[2])))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
history = lstm_model.fit(train_X_lstm, train_y, epochs=32, batch_size=24,
                         validation_data=(val_X_lstm, val_y))


# Make predictions
lstm_predict = lstm_model.predict(test_X_lstm, batch_size=24)
lr_predict = lr.predict(test_X)
svr_predict = svr.predict(test_X)


# Generate & print metrics
def generate_metrics(y_true, y_pred):
    return (
        f"MAE: {round(mean_absolute_error(y_true, y_pred), 9)}",
        f"MSE: {round(mean_squared_error(y_true, y_pred), 9)}",
        f"RMSE: {round(sqrt(mean_squared_error(y_true, y_pred)), 9)}",
        f"R2: {round(r2_score(y_true, y_pred), 9)}",
        f"EV: {round(explained_variance_score(y_true, y_pred), 9)}"
    )


lstm_metrics = generate_metrics(test_y, lstm_predict)
lr_metrics = generate_metrics(test_y, lr_predict)
svr_metrics = generate_metrics(test_y, svr_predict)

print("\nMetrics:")
print(f"LSTM:\t{lstm_metrics}")
print(f"LR:\t{lr_metrics}")
print(f"SVR:\t{svr_metrics}")

# Plot actual and predictions, loss history
fig, ax = plt.subplots(2)
ax[0].plot(lstm_predict[:, -1], label='LSTM prediction')
ax[0].plot(svr_predict, label='SVR prediction')
ax[0].plot(lr_predict, label='LR prediction')
ax[0].plot(test_y, label='actual')
ax[0].legend(loc='upper right')
ax[1].plot(history.history['loss'], label='train_loss')
ax[1].plot(history.history['val_loss'], label='val_loss')
ax[1].legend(loc='upper right')
plt.show()
