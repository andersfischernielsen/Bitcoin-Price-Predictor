import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Read
data = read_csv("D1.csv", header=0, index_col=0, skiprows=14)
values = data.values.astype('float64')

# Split data into test, train and validation with validation as most recent data
split = int(values.shape[0]*0.1)
train = values[:split*8, :]
val = values[split*8:split*9, :]
test = values[split*9:split*10, :]

# Scale data
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)
test = scaler.fit_transform(test)

# Split data into val and training
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# Reshape input to [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Training shapes
print("Training shapes:")
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)

# Instantiate model
model = Sequential()
units = int(train_X.shape[0]/10)
model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit network
history = model.fit(train_X, train_y, epochs=16, batch_size=24,
                    validation_data=(val_X, val_y), shuffle='batch')

# Plot loss history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Make a prediction
predict = model.predict(test_X, batch_size=24)

# Plot actual and prediction
plt.plot(predict[:, -1], label='predict')
plt.plot(test_y, label='actual')
plt.legend()
plt.show()
